import torch
import torch.nn as nn
import torch.nn.functional as F

from .loftr_module.transformer import LocalFeatureTransformer, DepthFeatureTransformer, FeatureFusionTransformer


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

    def __init__(self, config):
        super(CIM, self).__init__()

        self.block_dims = config["backbone"]["block_dims"]
        self.drop = config["fine"]["droprate"]
        self.use_inject = config["use_inject"]
        self.use_fusion = config["use_fusion"]

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
        if self.use_inject:
            self.depth_injector = DepthFeatureInjection(in_dim=384, out_dim=256, config=config["depth_injection"])
        if self.use_fusion:
            self.depth_fusion = DepthFeatureFusion(config)

    def forward(self, ms_feats, mask_c0=None, mask_c1=None):
        if isinstance(ms_feats, dict) and "rgb" in ms_feats and "depth" in ms_feats:
            rgb_feats = ms_feats["rgb"]
            depth_feats = ms_feats["depth"]
            if self.use_inject:
                # Inject depth features to image feature(1/16)
                depth0, depth1 = depth_feats
                depth0, depth1 = self.depth_injector(depth0, depth1, mask_c0, mask_c1)
                f8, f16, f32 = rgb_feats
                # print('depth0 shape:', depth0.shape, 'depth1 shape:', depth1.shape)
                # print('f8 shape:', rgb_feats[0].shape, 'f16 shape:', rgb_feats[1].shape, 'f32 shape:', rgb_feats[2].shape)
                # Inject depth features into F16
                # f16 = f16 + depth0
                depth_cat = torch.cat([depth0, depth1], dim=0)
                f16 = f16 + depth_cat
                ms_feats = (f8, f16, f32)

        if len(rgb_feats) == 3:  # same image shape
            f8, f16, f32 = rgb_feats
            f32 = self.fc32(f32)

            f32_0, f32_1 = f32.chunk(2, dim=0)
            f32_0, f32_1 = self.loftr_32(f32_0, f32_1, mask_c0, mask_c1)
            f32 = torch.cat([f32_0, f32_1], dim=0)

            f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
            # Fuse depth features into F32->F16
            if self.use_fusion:
                depth0, depth1 = depth_feats
                # Apply depth fusion to the upscaled f32 feature
                f32_up_0, f32_up_1 = f32_up.chunk(2, dim=0)
                f32_up_0, f32_up_1 = self.depth_fusion(f32_up_0, f32_up_1, depth0, depth1, mask_c0, mask_c1)
                f32_up = torch.cat([f32_up_0, f32_up_1], dim=0)
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

        elif len(rgb_feats) == 6:  # diffirent image shape
            f8_0, f16_0, f32_0, f8_1, f16_1, f32_1 = rgb_feats
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
    
class DepthFeatureFusion(nn.Module):
    """
    Feature Fusion Module for DepthAnythingV2 + EDM
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.depth_layer = DepthFeatureTransformer(config["depth_fusion"], is16=True)
        self.fusion_layer = FeatureFusionTransformer(config["depth_fusion"], is16=True)
        self.pre_extracted_depth = config["pre_extracted_depth"]
        self.depth_proj = nn.Conv2d(384, 256, kernel_size=1)

    def forward(self, feat_c0, feat_c1, depth0, depth1, mask_c0=None, mask_c1=None):
        if self.pre_extracted_depth:
            depth0 = depth0.permute(0, 2, 1).reshape(-1, 384, 37, 37)
            depth1 = depth1.permute(0, 2, 1).reshape(-1, 384, 37, 37)
        else:
            # Remove cls token when feature is extracted while training
            depth0 = depth0[:,1:].permute(0, 2, 1).reshape(-1, 384, 37, 37)
            depth1 = depth1[:,1:].permute(0, 2, 1).reshape(-1, 384, 37, 37)

        # Project depth features to match image feature dimensions
        depth0 = self.depth_proj(depth0)  # [B, 384, 37, 37]
        depth1 = self.depth_proj(depth1)  # -> [B, 256, 37, 37]

        # Resize depth features to match image feature spatial dimensions
        target_size = feat_c0.shape[2:]  # Get H, W from feat_c0
        depth0 = F.interpolate(depth0, size=target_size, mode='bilinear', align_corners=False)
        depth1 = F.interpolate(depth1, size=target_size, mode='bilinear', align_corners=False)
        
        # Depth feature cross-attention
        depth0, depth1 = self.depth_layer(depth0, depth1, mask_c0, mask_c1)

        # Fuse depth and image features
        fused_c0, fused_c1 = self.fusion_layer(
            feat_c0, feat_c1,  # Query: image features
            depth0, depth1,  # Key/Value: depth features
            mask_c0, mask_c1
        )
        return fused_c0, fused_c1

class DepthFeatureInjection(nn.Module):
    """Inject DepthAnythingV2 features into EDM via cross-attention or projection"""

    def __init__(self, in_dim, out_dim, config, cross_attn=True):
        super().__init__()
        self.cross_attn = cross_attn
        # in_dim: 384; from depth anything v2
        # out_dim: 256; from resnet18
        self.proj0 = Conv2d_BN_Act(in_dim, out_dim, ks=1)
        self.proj1 = Conv2d_BN_Act(in_dim, out_dim, ks=1)
        self.out_dim = out_dim
        self.pre_extracted_depth = config["pre_extracted_depth"]

        if cross_attn:
            self.attn = LocalFeatureTransformer(config["depth_injection"],
                                                is16=True)

    def forward(self, depth0, depth1, mask0=None, mask1=None):
        # depth0, depth1: [N, 1369, 384] Fixed
        # print(depth0)
        # print(depth1)
        if self.pre_extracted_depth:
            depth0 = depth0.permute(0, 2, 1).reshape(-1, 384, 37, 37)
            depth1 = depth1.permute(0, 2, 1).reshape(-1, 384, 37, 37)
        else:
            # Remove cls token when feature is extracted while training
            depth0 = depth0[:,1:].permute(0, 2, 1).reshape(-1, 384, 37, 37)
            depth1 = depth1[:,1:].permute(0, 2, 1).reshape(-1, 384, 37, 37)
        # Interpolate to F16
        depth0 = F.interpolate(depth0, (52, 52), mode="bilinear")
        depth1 = F.interpolate(depth1, (52, 52), mode="bilinear")
        # Project 384 to 256(1/16)
        d0 = self.proj0(depth0)
        d1 = self.proj1(depth1)
        # print('d0 shape:', d0, 'd1 shape:', d1)

        if self.cross_attn:
            # print('mask0 shape:', mask0, 'mask1 shape:', mask1)
            d0, d1 = self.attn(d0, d1, mask0, mask1)
        # print(d0, d1, depth0, depth1)
        # print('d0', d0, 'd1', d1, 'depth0', depth0, 'depth1', depth1)
        return d0, d1

class DepthAnythingFeatureExtractor(nn.Module):
    """Wraps DepthAnythingV2 to extract the final feature map (not depth map)."""

    def __init__(self, config, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
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
