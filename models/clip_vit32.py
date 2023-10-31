import open_clip
from typing import Tuple, Callable, Dict, Any
from torch import Tensor, nn
from PIL import Image


def get_clip_vit32_resources() -> Tuple[nn.Module, Callable[[Image.Image], Tensor], Dict[str, str]]:
    """
    Gets all resources for RDM extraction for CLIP ViT-B/32.

    Returns:
        A tuple of the model, preprocess function and layers names.
    """
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    
    vit_layers = {'Input': 'input_1'} # Input images
    vit_layers = vit_layers | {f'visual.transformer.resblocks.{i}': f'visual.transformer.resblocks.{i}' for i in range(12)} # All resblocks for CLIP ViT-B/32
    vit_layers = vit_layers | {'Output': 'matmul_1_426'} # Output embeddings

    return model.visual, preprocess, vit_layers
