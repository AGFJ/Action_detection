import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from functools import partial

__all__ = ['resnext50', 'resnext101']

model_urls = {
    "resnext50": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/resnext-50-kinetics.pth",
    "resnext101": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/resnext-101-kinetics.pth",
    "resnext152": "https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/resnext-152-kinetics.pth"
}


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()

    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    zero_pads = zero_pads.to(out.data.device)
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)

        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)

        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP   """

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv3d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv3d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv3d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False),  # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False),  # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()  # 输入x
        attn = self.conv0(x)  # depth-wise conv
        attn = self.conv_spatial(attn)  # depth-wise dilation convolution
        f_x = self.conv1(attn)  # 1x1 conv        #静态部分输出
        # append a SE operation
        b, c, _, _, _ = x.size()
        se_atten = self.max_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1, 1)  # 动态部分输出

        out = se_atten * f_x * u

        return out  # TAU整体公式


class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    # d_model即输入或输出通道数
    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)  # 1x1 conv
        self.activation = nn.GELU()  # GELU
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)  # TAM
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()  # 拷贝
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 cardinality=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)

        self.TAU = TemporalAttention(d_model=256)

        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)

        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)

        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.TAU(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if x.size(2) > 1:
            x = torch.mean(x, dim=2, keepdim=True)

        return x


def load_weight(model, arch):
    print('Loading pretrained weight ...')
    # checkpoint state dict
    url = model_urls[arch]
    checkpoint = load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
    checkpoint_state_dict = checkpoint.pop('state_dict')

    # model state dict
    model_state_dict = model.state_dict()
    # reformat checkpoint_state_dict:
    new_state_dict = {}
    for k in checkpoint_state_dict.keys():
        v = checkpoint_state_dict[k]
        new_state_dict[k[7:]] = v

    # check
    for k in list(new_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(new_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                new_state_dict.pop(k)
                # print(k)
        else:
            new_state_dict.pop(k)
            # print(k)

    model.load_state_dict(new_state_dict,strict=False)

    return model


def resnext50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext50')

    return model


def resnext101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext101')

    return model


# build 3D resnet
def build_resnext_3d(model_name='resnext50', pretrained=False):
    if model_name == 'resnext50':
        model = resnext50(pretrained=pretrained)
        feat = 2048

    elif model_name == 'resnext101':
        model = resnext101(pretrained=pretrained)
        feat = 2048

    return model, feat


if __name__ == '__main__':
    import time

    model, feat = build_resnext_3d(model_name='resnext50', pretrained=True)
    print(model)
    print(feat)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    x = torch.randn(4, 3, 16, 32, 32).to(device)
    print(x.size())
    for i in range(10):
        # star time
        t0 = time.time()
        y = model(x).squeeze(2)
        # print('time', time.time() - t0)

    print(y.size())
