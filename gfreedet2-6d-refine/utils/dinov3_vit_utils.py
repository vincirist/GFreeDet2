import torch
import torch.nn as nn
from modelscope import AutoModel
from typing import Dict, List, Union, Callable
from torchvision import transforms as T
from torch.nn import functional as F


class Dinov3FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_layer: int = -1,
        apply_norm: bool = True,
        out_bchw: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):

        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name
        self.apply_norm = apply_norm

        self.model = (
            AutoModel.from_pretrained(
                model_name,
                device_map=self.device,
            )
            .to(self.device)
            .eval()
        )

        self.n_registers = self.model.config.num_register_tokens
        self.patch_size = self.model.config.patch_size
        self.stride = self.model.embeddings.patch_embeddings.stride[0]

        self.out_bchw = out_bchw

        self.target_layer = self._validate_layers(target_layer)
        self.hooks = []
        self.feature_maps = {}

        self._register_hooks()

        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def _validate_layers(self, idx: int) -> int:

        total_layers = len(self.model.layer)

        if idx < 0:
            idx = total_layers + idx
            if not (0 <= idx < total_layers):
                raise ValueError(f"层索引 {idx} 越界，模型共 {total_layers} 层")
        return idx

    def _hook_fn(self, layer_idx: int) -> Callable:

        def hook(module, input, output):
            self.feature_maps[layer_idx] = output.detach()

        return hook

    def _register_hooks(self):

        for idx in [self.target_layer]:
            layer = self.model.layer[idx]
            hook = layer.register_forward_hook(self._hook_fn(idx))
            self.hooks.append(hook)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:

        self.feature_maps = {}

        inputs = self.normalize(images)

        with torch.no_grad():
            _ = self.model(inputs)

        feat = self.feature_maps[self.target_layer]
        if self.apply_norm:
            feat = self.model.norm(feat)

        cls_tokens = feat[:, 0]
        patch_tokens = feat[:, 1 + 4 :]

        if self.out_bchw:
            bsz, _, h, w = images.shape
            d = patch_tokens.shape[-1]
            num_patches = (
                1 + (h - self.patch_size) // self.stride,
                1 + (w - self.patch_size) // self.stride,
            )
            feature_maps = patch_tokens.reshape(bsz, num_patches[0], num_patches[1], d).permute(0, 3, 1, 2)
        else:
            feature_maps = patch_tokens

        return {
            "cls_tokens": cls_tokens,  # BxD
            "feature_maps": feature_maps,  # BxDxHxW
        }

    def __del__(self):

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_layer_info(self) -> List[str]:

        return [f"vit layer {i}" for i in range(len(self.model.layer))]


if __name__ == "__main__":
    import imageio.v2 as iio
    import cv2
    import numpy as np

    # model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"  # 12 layers, dim 384
    # model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"  # 12 layers, dim 384
    # model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # 12 layers, dim 768
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # 24 layers, dim 1024
    # model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"  # 32 layers, dim 1280
    # model_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"  # 40 layers, dim 4096
    if "vit7b16" in model_name:
        device = "cpu"
    else:
        device = "cuda"
    extractor = Dinov3FeatureExtractor(
        model_name=model_name,
        target_layer=-1,
        device=device,
    )

    image_path = "local_data/bop_datasets/hopev2/test/000001/rgb/000000.png"
    image = iio.imread(image_path)
    image = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

    image_tensors = torch.cat([image_tensor] * 2)

    outputs = extractor(image_tensors)
    print("layer info:", extractor.get_layer_info())
    for k, v in outputs.items():
        print(k, v.shape)
    import ipdb

    ipdb.set_trace()
