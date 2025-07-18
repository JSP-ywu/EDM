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


class CIM(nn.Module):
    """Feature Aggregation, Correlation Injection Module"""

    def __init__(self, config, depth_injector=None):
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
        self.depth_injector = depth_injector

    def forward(self, ms_feats, mask_c0=None, mask_c1=None):
        if isinstance(ms_feats, dict) and "rgb" in ms_feats and "depth" in ms_feats:
            rgb_feats = ms_feats["rgb"]
            depth_feats = ms_feats["depth"]
            if self.depth_injector is not None:
                depth0, depth1 = depth_feats
                depth0, depth1 = self.depth_injector(depth0, depth1, mask_c0, mask_c1)
                f8, f16, f32 = rgb_feats
                f8 = f8 + depth0  # inject at 1/8 resolution
                ms_feats = (f8, f16, f32)

        if len(ms_feats) == 3:  # same image shape
            f8, f16, f32 = ms_feats
            f32 = self.fc32(f32)

            f32_0, f32_1 = f32.chunk(2, dim=0)
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

class DepthFeatureInjection(nn.Module):
    """Inject DepthAnythingV2 features into EDM via cross-attention or projection"""

    def __init__(self, in_dim, out_dim, config, cross_attn=True):
        super().__init__()
        self.cross_attn = cross_attn

        self.proj0 = Conv2d_BN_Act(in_dim, out_dim, ks=1)
        self.proj1 = Conv2d_BN_Act(in_dim, out_dim, ks=1)

        if cross_attn:
            self.attn = LocalFeatureTransformer(config["depth_injection"])

    def forward(self, depth0, depth1, mask0=None, mask1=None):
        # depth0, depth1: [N, C_in, H, W]
        d0 = self.proj0(depth0)
        d1 = self.proj1(depth1)

        if self.cross_attn:
            d0, d1 = self.attn(d0, d1, mask0, mask1)

        return d0 + depth0, d1 + depth1  # residual fusion

class DepthAnythingFeatureExtractor(nn.Module):
    """Wraps DepthAnythingV2 to extract the final feature map (not depth map)."""

    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_name,
                                                            trust_remote_code=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name,
                                                                 trust_remote_code=True)
        self.model.eval()  # important for inference stability

    @torch.no_grad()
    def forward(self, image0, image1):
        """
        image0, image1: torch.Tensor, shape [B, 3, H, W], values in [0, 1]
        Returns: depth_feat0, depth_feat1: [B, C, H/4, W/4]
        """
        # HuggingFace expects images in [0, 255] and shape HWC
        import torchvision.transforms.functional as TF
        # print(f"[DEBUG] image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"[DEBUG] image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
        # for i, img in enumerate(image0 + image1):
        #     print(f"[DEBUG] Image {i} shape: {img.shape}, dtype: {img.dtype}")
        #     if isinstance(img, torch.Tensor) and img.ndim != 3:
        #         print(f"[WARNING] Bad shape: {img.shape} — skipping")
        image0 = [TF.to_pil_image(img.cpu()) for img in image0]
        image1 = [TF.to_pil_image(img.cpu()) for img in image1]
        inputs = self.processor(images=image0 + image1, return_tensors="pt")
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        B = len(image0)
        return outputs.hidden_states[-1][:B], outputs.hidden_states[-1][B:]
