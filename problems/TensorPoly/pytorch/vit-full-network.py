import torch


class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = torch.zeros((batch_size, self.num_patches, self.embed_dim))
        x = torch.cat([
            torch.zeros((batch_size, 1, self.embed_dim)),
            x
        ], dim=1)

        x = x + torch.zeros((1, self.num_patches + 1, self.embed_dim))

        for _ in range(self.depth):
            x = x + torch.zeros_like(x)

        logits = torch.zeros((batch_size, self.num_classes))
        return logits
