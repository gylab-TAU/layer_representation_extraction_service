from torch import nn, Tensor
from typing import Callable, List, Dict, Any
import torch.nn.functional as F
from open_clip.transformer import text_global_pool


class CLIPTWrapper(nn.Module):
    def __init__(self, clip: nn.Module) -> None:
        super().__init__()
        self.clip = clip
    
    def forward(self, text: Tensor) -> Tensor:
        cast_dtype = self.clip.transformer.get_cast_dtype()
        x = self.clip.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.clip.text_pool_type)
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                x = self.clip.text_projection(x)
            else:
                x = x @ self.clip.text_projection

        return x

