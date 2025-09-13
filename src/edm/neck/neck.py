import torch
import torch.nn as nn
import torch.nn.functional as F

from .loftr_module.transformer import LocalFeatureTransformer


class Conv2d_BN_Act(nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act=None,
        drop=None,
    ):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module(
            "c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        if act != None:
            self.add_module("a", act)
        if drop != None:
            self.add_module("d", nn.Dropout(drop))

class DeferredConv1x1(nn.Module):
    def __init__(self, out_ch: int, bias: bool):
        super().__init__()
        self.out_ch = out_ch
        self.bias = bias
        self.conv = None
    def forward(self, x):
        if self.conv is None:
            in_ch = x.shape[1]
            conv = nn.Conv2d(in_ch, self.out_ch, kernel_size=1, bias=self.bias)
            conv.to(dtype=x.dtype, device=x.device)
            self.conv = conv
        return self.conv(x)

class CIM(nn.Module):
    """Feature Aggregation, Correlation Injection Module"""

    def __init__(self, config):
        super(CIM, self).__init__()

        self.block_dims = config["backbone"]["block_dims"]
        self.drop = config["fine"]["droprate"]

        self.fc32 = Conv2d_BN_Act(
            self.block_dims[-1], self.block_dims[-1], 1, drop=self.drop
        )
        self.fc16 = Conv2d_BN_Act(
            self.block_dims[-2], self.block_dims[-1], 1, drop=self.drop
        )
        self.fc8 = Conv2d_BN_Act(
            self.block_dims[-3], self.block_dims[-1], 1, drop=self.drop
        )
        self.att32 = Conv2d_BN_Act(
            self.block_dims[-1],
            self.block_dims[-1],
            1,
            act=nn.Sigmoid(),
            drop=self.drop,
        )
        self.att16 = Conv2d_BN_Act(
            self.block_dims[-1],
            self.block_dims[-1],
            1,
            act=nn.Sigmoid(),
            drop=self.drop,
        )
        self.dwconv16 = nn.Sequential(
            Conv2d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                ks=3,
                pad=1,
                groups=self.block_dims[-1],
                act=nn.GELU(),
            ),
            Conv2d_BN_Act(self.block_dims[-1], self.block_dims[-1], 1),
        )
        self.dwconv8 = nn.Sequential(
            Conv2d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                ks=3,
                pad=1,
                groups=self.block_dims[-1],
                act=nn.GELU(),
            ),
            Conv2d_BN_Act(self.block_dims[-1], self.block_dims[-1], 1),
        )

        self.loftr_32 = LocalFeatureTransformer(config["neck"])

        # --- Optional hidden-state (e.g., Depth Anything v2) injection ---
        self.hidden_use_train_only = config["fine"].get("use_hidden_train_only", True)
        self.hidden_fuse = config["fine"].get("hidden_fuse", "film")  # 'film' or 'add'
        self.hidden_weight = float(config["fine"].get("hidden_weight", 0.1))

        out_ch = self.block_dims[-1]
        self.hid_proj = DeferredConv1x1(out_ch, bias=False)
        self.hid_gamma = DeferredConv1x1(out_ch, bias=True)
        self.hid_beta  = DeferredConv1x1(out_ch, bias=True)

    def forward(self, ms_feats, mask_c0=None, mask_c1=None,
                hidden0=None, hidden1=None, inject_hidden=False):
        if len(ms_feats) == 3:  # same image shape
            f8, f16, f32 = ms_feats
            f32 = self.fc32(f32)

            f32_0, f32_1 = f32.chunk(2, dim=0)

            # --- Optional hidden-state injection (train-time only) ---
            if inject_hidden:
                if hidden0 is not None and hidden1 is not None:
                    h0 = F.interpolate(hidden0.detach(), size=f32_0.shape[-2:], mode="bilinear", align_corners=False)
                    h1 = F.interpolate(hidden1.detach(), size=f32_1.shape[-2:], mode="bilinear", align_corners=False)
                    if self.hidden_fuse == "film":
                        g0 = torch.tanh(self.hid_gamma(h0)) * self.hidden_weight
                        b0 = torch.tanh(self.hid_beta(h0))  * self.hidden_weight
                        g1 = torch.tanh(self.hid_gamma(h1)) * self.hidden_weight
                        b1 = torch.tanh(self.hid_beta(h1))  * self.hidden_weight
                        f32_0 = f32_0 * (1.0 + g0) + b0
                        f32_1 = f32_1 * (1.0 + g1) + b1
                    else:
                        f32_0 = f32_0 + self.hidden_weight * self.hid_proj(h0)
                        f32_1 = f32_1 + self.hidden_weight * self.hid_proj(h1)

            f32_0, f32_1 = self.loftr_32(f32_0, f32_1, mask_c0, mask_c1)
            f32 = torch.cat([f32_0, f32_1], dim=0)

            f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
            att32_up = F.interpolate(self.att32(
                f32), scale_factor=2.0, mode="bilinear")
            f16 = self.fc16(f16)
            f16 = self.dwconv16(f16 * att32_up + f32_up)
            f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
            att16_up = F.interpolate(self.att16(
                f16), scale_factor=2.0, mode="bilinear")
            f8 = self.fc8(f8)
            f8 = self.dwconv8(f8 * att16_up + f16_up)

            feat_c0, feat_c1 = f8.chunk(2)

        elif len(ms_feats) == 6:  # diffirent image shape
            f8_0, f16_0, f32_0, f8_1, f16_1, f32_1 = ms_feats
            f32_0 = self.fc32(f32_0)
            f32_1 = self.fc32(f32_1)

            if inject_hidden:
                if hidden0 is not None:
                    h0 = F.interpolate(hidden0.detach(), size=f32_0.shape[-2:], mode="bilinear", align_corners=False)
                    if self.hidden_fuse == "film":
                        g0 = torch.tanh(self.hid_gamma(h0)) * self.hidden_weight
                        b0 = torch.tanh(self.hid_beta(h0))  * self.hidden_weight
                        f32_0 = f32_0 * (1.0 + g0) + b0
                    else:
                        f32_0 = f32_0 + self.hidden_weight * self.hid_proj(h0)
                if hidden1 is not None:
                    h1 = F.interpolate(hidden1.detach(), size=f32_1.shape[-2:], mode="bilinear", align_corners=False)
                    if self.hidden_fuse == "film":
                        g1 = torch.tanh(self.hid_gamma(h1)) * self.hidden_weight
                        b1 = torch.tanh(self.hid_beta(h1))  * self.hidden_weight
                        f32_1 = f32_1 * (1.0 + g1) + b1
                    else:
                        f32_1 = f32_1 + self.hidden_weight * self.hid_proj(h1)


            f32_0, f32_1 = self.loftr_32(f32_0, f32_1, mask_c0, mask_c1)

            f8, f16, f32 = f8_0, f16_0, f32_0
            f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
            att32_up = F.interpolate(self.att32(
                f32), scale_factor=2.0, mode="bilinear")
            f16 = self.fc16(f16)
            f16 = self.dwconv16(f16 * att32_up + f32_up)
            f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
            att16_up = F.interpolate(self.att16(
                f16), scale_factor=2.0, mode="bilinear")
            f8 = self.fc8(f8)
            f8 = self.dwconv8(f8 * att16_up + f16_up)
            feat_c0 = f8

            f8, f16, f32 = f8_1, f16_1, f32_1
            f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
            att32_up = F.interpolate(self.att32(
                f32), scale_factor=2.0, mode="bilinear")
            f16 = self.fc16(f16)
            f16 = self.dwconv16(f16 * att32_up + f32_up)
            f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
            att16_up = F.interpolate(self.att16(
                f16), scale_factor=2.0, mode="bilinear")
            f8 = self.fc8(f8)
            f8 = self.dwconv8(f8 * att16_up + f16_up)
            feat_c1 = f8

        return feat_c0, feat_c1
