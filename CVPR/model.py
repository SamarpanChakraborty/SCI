import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import LossFunction


class GuidedFilter(nn.Module):
    """Edge-preserving denoise on the SCI reflectance output.

    Why: r = input / illu amplifies sensor noise in dark regions. The training
    loss only smooths illu, never r, so noise survives. A self-guided filter
    removes it without learnable parameters and runs in O(N) via box filters.
    """

    def __init__(self, radius=4, eps=1e-3):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def _box(self, x):
        r = self.radius
        x = F.pad(x, (r, r, r, r), mode='replicate')
        return F.avg_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=0)

    def forward(self, p, guide=None):
        I = p if guide is None else guide
        mean_I = self._box(I)
        mean_p = self._box(p)
        mean_Ip = self._box(I * p)
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = self._box(I * I)
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self._box(a)
        mean_b = self._box(b)
        return mean_a * I + mean_b


def _fuse_conv_bn(conv, bn):
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True,
    )
    w = conv.weight.detach().clone()
    b = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(
        conv.out_channels, device=w.device, dtype=w.dtype)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data = w * scale.view(-1, 1, 1, 1)
    fused.bias.data = (b - bn.running_mean) * scale + bn.bias
    return fused.to(w.device)


def _fuse_sequential(seq):
    layers = list(seq.children())
    out = []
    i = 0
    while i < len(layers):
        if (i + 1 < len(layers)
                and isinstance(layers[i], nn.Conv2d)
                and isinstance(layers[i + 1], nn.BatchNorm2d)):
            out.append(_fuse_conv_bn(layers[i], layers[i + 1]))
            i += 2
        else:
            out.append(layers[i])
            i += 1
    return nn.Sequential(*out)



class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta



class Network(nn.Module):

    def __init__(self, stage=3):
        super(Network, self).__init__()
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):

        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.enhance(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            att = self.calibrate(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss



class Finetunemodel(nn.Module):

    def __init__(self, weights, denoise=False, denoise_radius=4, denoise_eps=1e-3):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self._criterion = LossFunction()
        self.guided = GuidedFilter(denoise_radius, denoise_eps) if denoise else None

        base_weights = torch.load(weights, map_location='cpu')
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def fuse(self):
        """Fold BatchNorm into the preceding Conv2d. Inference only; call after eval()."""
        self.eval()
        enh = self.enhance
        enh.in_conv = _fuse_sequential(enh.in_conv)
        enh.out_conv = _fuse_sequential(enh.out_conv)
        seen = {}
        new_blocks = nn.ModuleList()
        for block in enh.blocks:
            key = id(block)
            if key not in seen:
                seen[key] = _fuse_sequential(block)
            new_blocks.append(seen[key])
        enh.blocks = new_blocks
        return self

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        if self.guided is not None:
            r = self.guided(r)
            r = torch.clamp(r, 0, 1)
        return i, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss

