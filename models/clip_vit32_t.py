import open_clip
from typing import Tuple, Callable, Dict, List, Any
from torch import Tensor, nn
from .clip_t_wrapper import CLIPTWrapper


def get_clip_vit32_text_resources() -> Tuple[nn.Module, Callable[[List[str]], Tensor], Dict[str, Any]]:
    """
    Gets all resources for RDM extraction for CLIP ViT-B/32.

    Returns:
        A tuple of the model, preprocess function and layers names.
    """
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = CLIPTWrapper(model)
    model.eval()
    
    vit_layers = {'Input': 'clip.token_embedding'} # Input images
    vit_layers = vit_layers | {f'clip.transformer.resblocks.{i}': f'clip.transformer.resblocks.{i}' for i in range(12)} # All resblocks for CLIP ViT-B/32
    vit_layers = vit_layers | {'Output': 'output_1'} #{'Output': 'matmul_1_443'} # Output embeddings

    return model, tokenizer, vit_layers
