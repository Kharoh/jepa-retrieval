"""Multimodal JEPA with proper attention-based patch-label interaction."""

from __future__ import annotations

from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torchvision.ops import MLP
from timm.models.vision_transformer import Mlp
from timm.layers import DropPath


class SIGReg(torch.nn.Module):
    """SIGReg regularization - reused from your existing code."""
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class LabelEmbedding(nn.Module):
    """Learnable embedding for class labels."""
    
    def __init__(self, num_classes: int = 10, embed_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)
        
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: (B,) tensor of class labels
        Returns:
            (B, 1, embed_dim) label embeddings ready to concat with patches
        """
        return self.embedding(labels).unsqueeze(1)  # (B, 1, embed_dim)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) images
        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultimodalViTEncoder(nn.Module):
    """Custom ViT encoder with integrated label token support.
    
    Label tokens are added at the patch embedding stage and interact
    with patch tokens through all transformer blocks via self-attention.
    """
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_cls_token: bool = False,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token (optional, for compatibility with standard ViT)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
            
        # Label embedding
        self.label_embed = LabelEmbedding(num_classes, embed_dim)
        
        # Positional embeddings
        # +1 for cls token if used, +1 for label token
        num_tokens = num_patches + (1 if use_cls_token else 0) + 1  # +1 for label
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks (custom to expose attention maps)
        self.blocks = nn.ModuleList([
            BlockWithAttn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(
        self, 
        x: torch.Tensor, 
        labels: torch.Tensor = None,
        include_label: bool = False
    ) -> torch.Tensor:
        """
        Forward through patch embedding and transformer blocks.
        
        Args:
            x: (B, C, H, W) images
            labels: (B,) label indices (required if include_label=True)
            include_label: Whether to include label token in sequence
        
        Returns:
            (B, N, embed_dim) where N = num_patches [+ 1 cls] [+ 1 label]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Build token sequence
        tokens = []
        
        # Add CLS token if used
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens.append(cls_tokens)
        
        # Add patch tokens
        tokens.append(x)
        
        # Add label token if requested
        if include_label:
            if labels is None:
                raise ValueError("labels required when include_label=True")
            label_tokens = self.label_embed(labels)  # (B, 1, embed_dim)
            tokens.append(label_tokens)
        
        # Concatenate all tokens
        x = torch.cat(tokens, dim=1)  # (B, N, embed_dim)
        
        # Add positional embedding
        # Note: pos_embed is sized for max tokens (patches + cls + label)
        # If label not included, we just use fewer positions
        pos_embed = self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x + pos_embed)
        
        # Apply transformer blocks (self-attention allows all tokens to interact)
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x

    def get_attention_maps(self) -> List[torch.Tensor]:
        """Return attention maps from the last forward pass.

        Returns:
            List of tensors with shape (B, heads, N, N) per block.
        """
        attn_maps = []
        for block in self.blocks:
            attn_maps.append(block.attn.attn_map)
        return attn_maps

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor = None,
        include_label: bool = False,
        return_all_tokens: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) images
            labels: (B,) label indices
            include_label: Whether to include label token
            return_all_tokens: If True, return all tokens; if False, return mean

        Returns:
            (B, N, embed_dim) or (B, embed_dim) depending on return_all_tokens
        """
        x = self.forward_features(x, labels, include_label)

        if return_all_tokens:
            return x
        else:
            # Global average pooling over all tokens
            return x.mean(dim=1)


class AttentionWithMap(nn.Module):
    """Self-attention that stores the last attention map."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_map = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_map = attn.detach()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithAttn(nn.Module):
    """Transformer block that exposes attention maps."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionWithMap(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultimodalViTLeJEPA(nn.Module):
    """ViT-based encoder with label token support for multimodal JEPA.
    
    This version uses a custom ViT where label tokens interact with patch
    tokens through self-attention in the transformer blocks.
    """
    
    def __init__(
        self,
        proj_dim: int,
        img_size: int,
        in_chans: int = 1,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        drop_path_rate: float = 0.1,
        use_cls_token: bool = False,
    ):
        super().__init__()
        
        # Custom ViT encoder with label token support
        self.backbone = MultimodalViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            use_cls_token=use_cls_token,
        )
        
        # Projection head
        self.proj = MLP(
            embed_dim, 
            [embed_dim * 4, embed_dim * 4, proj_dim], 
            norm_layer=nn.BatchNorm1d
        )
        
        self.in_chans = in_chans
        self.img_size = img_size
        self.emb_dim = embed_dim
        self.num_features = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
    def encode_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image patches only (no label token).
        
        Args:
            x: (B, V, D) flattened or (B*V, C, H, W) image tensor
        Returns:
            emb: (B, V, emb_dim) embeddings (mean over patches)
            proj: (B, V, proj_dim) projections
        """
        # Handle different input formats
        if x.dim() == 3:
            b, v, d = x.shape
            x = x.view(b * v, self.in_chans, self.img_size, self.img_size)
        elif x.dim() == 2:
            b = x.shape[0]
            v = 1
            x = x.view(b * v, self.in_chans, self.img_size, self.img_size)
        else:
            b = x.shape[0]
            v = 1
        
        # Forward without label token
        tokens = self.backbone(
            x, 
            labels=None, 
            include_label=False,
            return_all_tokens=True
        )  # (B*V, N, emb_dim) where N = num_patches (+ cls if used)
        
        # Pool over all tokens
        emb = tokens.mean(dim=1)  # (B*V, emb_dim)
        
        # Project
        proj = self.proj(emb)  # (B*V, proj_dim)
        
        # Reshape to (B, V, D)
        emb = emb.view(b, v, -1)
        proj = proj.view(b, v, -1)
        
        return emb, proj
    
    def encode_label(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode label tokens only (no image).
        
        Args:
            labels: (B,) label tensor
        Returns:
            emb: (B, emb_dim) embeddings
            proj: (B, proj_dim) projections
        """
        # Get label embedding directly
        emb = self.backbone.label_embed(labels).squeeze(1)  # (B, emb_dim)
        proj = self.proj(emb)  # (B, proj_dim)
        return emb, proj
    
    def encode_multimodal(
        self, 
        x: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image + label token together with attention interaction.
        
        The label token is added to the patch sequence and all tokens
        interact through self-attention in the transformer blocks.
        
        Args:
            x: (B, V, D) flattened or (B*V, C, H, W) image tensor
            labels: (B,) label tensor
        Returns:
            emb: (B, V, emb_dim) embeddings (mean over all tokens including label)
            proj: (B, V, proj_dim) projections
        """
        # Handle different input formats
        if x.dim() == 3:
            b, v, d = x.shape
            x = x.view(b * v, self.in_chans, self.img_size, self.img_size)
        elif x.dim() == 2:
            b = x.shape[0]
            v = 1
            x = x.view(b * v, self.in_chans, self.img_size, self.img_size)
        else:
            b = x.shape[0]
            v = 1
        
        # Expand labels to match batch*views
        labels_expanded = labels.repeat_interleave(v)  # (B*V,)
        
        # Forward with label token (attention interaction happens here!)
        tokens = self.backbone(
            x,
            labels=labels_expanded,
            include_label=True,
            return_all_tokens=True
        )  # (B*V, N+1, emb_dim) where N = num_patches (+ cls if used), +1 for label
        
        # Pool over all tokens (including label token)
        emb = tokens.mean(dim=1)  # (B*V, emb_dim)
        
        # Project
        proj = self.proj(emb)  # (B*V, proj_dim)
        
        # Reshape to (B, V, D)
        emb = emb.view(b, v, -1)
        proj = proj.view(b, v, -1)
        
        return emb, proj
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: torch.Tensor = None,
        encode_mode: str = "image_only"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with different encoding modes."""
        if encode_mode == "image_only":
            return self.encode_image(x)
        elif encode_mode == "label_only":
            if labels is None:
                raise ValueError("labels required for label_only mode")
            return self.encode_label(labels)
        elif encode_mode == "multimodal":
            if labels is None:
                raise ValueError("labels required for multimodal mode")
            return self.encode_multimodal(x, labels)
        else:
            raise ValueError(f"Unknown encode_mode: {encode_mode}")


class MultimodalLeJEPAModel(nn.Module):
    """Multimodal LeJEPA with label tokens and SIGReg regularization.
    
    Label tokens interact with patch tokens through self-attention.
    """
    
    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 192,
        proj_dim: int = 64,
        num_classes: int = 10,
        lamb: float = 0.5,
        use_vit: bool = True,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        vit_backbone: str = "vit_tiny_patch16_224",
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        drop_path_rate: float = 0.1,
        use_cls_token: bool = False,
    ):
        super().__init__()
        
        if not use_vit:
            raise NotImplementedError("Only ViT encoder supported for multimodal JEPA")
            
        _, height, width = image_shape
        if height != width:
            raise ValueError("ViT backbone requires square images")
        
        # Use custom multimodal ViT encoder
        self.encoder = MultimodalViTLeJEPA(
            proj_dim=proj_dim,
            img_size=height,
            in_chans=image_shape[0],
            num_classes=num_classes,
            embed_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,
            use_cls_token=use_cls_token,
        )
        
        self.sigreg = SIGReg(knots=17)
        self.lamb = lamb
        self.multimodal_mode = "attention"  # Always uses attention-based interaction
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass - defaults to image-only encoding."""
        return self.encoder.encode_image(x)
        
    def compute_loss(
        self, 
        x: torch.Tensor, 
        labels: torch.Tensor = None,
        use_multimodal: bool = False,
        label_drop_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute LeJEPA loss with optional multimodal training.
        
        Args:
            x: (B, V, D) augmented views
            labels: (B,) labels (required if use_multimodal=True)
            use_multimodal: Whether to include label tokens in training
            label_drop_prob: Probability of dropping label token per view
        """
        if use_multimodal:
            if labels is None:
                raise ValueError("labels required for multimodal training")
            if label_drop_prob > 0.0:
                if x.dim() == 2:
                    x_views = x.unsqueeze(1)
                elif x.dim() == 3:
                    x_views = x
                else:
                    raise ValueError("Expected x to have shape (B, V, D) or (B, D)")

                b, v, _ = x_views.shape
                drop_mask = torch.rand((b, v), device=x_views.device) < label_drop_prob

                if drop_mask.all():
                    emb, proj = self.encoder.encode_image(x_views)
                elif (~drop_mask).all():
                    emb, proj = self.encoder.encode_multimodal(x_views, labels)
                else:
                    emb_views = []
                    proj_views = []

                    for view_idx in range(v):
                        view_x = x_views[:, view_idx:view_idx + 1, :]
                        view_mask = drop_mask[:, view_idx]

                        if view_mask.all():
                            emb_view, proj_view = self.encoder.encode_image(view_x)
                        elif (~view_mask).all():
                            emb_view, proj_view = self.encoder.encode_multimodal(view_x, labels)
                        else:
                            idx_keep = (~view_mask).nonzero(as_tuple=False).squeeze(1)
                            idx_drop = view_mask.nonzero(as_tuple=False).squeeze(1)

                            emb_view = None
                            proj_view = None

                            if idx_keep.numel() > 0:
                                emb_keep, proj_keep = self.encoder.encode_multimodal(
                                    view_x[idx_keep], labels[idx_keep]
                                )
                                emb_view = torch.zeros(
                                    (b, 1, emb_keep.size(-1)),
                                    device=view_x.device,
                                    dtype=emb_keep.dtype,
                                )
                                proj_view = torch.zeros(
                                    (b, 1, proj_keep.size(-1)),
                                    device=view_x.device,
                                    dtype=proj_keep.dtype,
                                )
                                emb_view[idx_keep] = emb_keep
                                proj_view[idx_keep] = proj_keep

                            if idx_drop.numel() > 0:
                                emb_drop, proj_drop = self.encoder.encode_image(view_x[idx_drop])
                                if emb_view is None:
                                    emb_view = torch.zeros(
                                        (b, 1, emb_drop.size(-1)),
                                        device=view_x.device,
                                        dtype=emb_drop.dtype,
                                    )
                                    proj_view = torch.zeros(
                                        (b, 1, proj_drop.size(-1)),
                                        device=view_x.device,
                                        dtype=proj_drop.dtype,
                                    )
                                emb_view[idx_drop] = emb_drop
                                proj_view[idx_drop] = proj_drop

                        emb_views.append(emb_view)
                        proj_views.append(proj_view)

                    emb = torch.cat(emb_views, dim=1)
                    proj = torch.cat(proj_views, dim=1)
            else:
                emb, proj = self.encoder.encode_multimodal(x, labels)
        else:
            emb, proj = self.encoder.encode_image(x)
        
        # LeJEPA loss: minimize variance across views
        proj_mean = proj.mean(dim=1, keepdim=True)
        inv_loss = (proj_mean - proj).square().mean()
        
        # SIGReg regularization
        sigreg_loss = self.sigreg(proj.flatten(0, 1))
        
        total_loss = self.lamb * sigreg_loss + (1 - self.lamb) * inv_loss
        
        return {
            "total": total_loss,
            "inv": inv_loss,
            "sigreg": sigreg_loss,
            "emb": emb,
            "proj": proj,
        }