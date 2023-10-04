from torch import nn, Tensor
import torch
import torchlens as tl
import torchmetrics.functional as F
from typing import Callable, List, Dict
from PIL import Image
import numpy as np


def balanced_RDMs(model: nn.Module, preprocess: Callable[[Image.Image], Tensor], imgs_paths: List[str], layers_names: Dict[str, str]) -> Dict[str, np.ndarray]:
    '''
    Calculate RDMs for a given model and images.
    Parallelizing the calculation over GPU using batch processing (batch size is the number of images).

    Args:
        model: the model to calculate RDMs for.
        preprocess: a function that preprocesses an image to a tensor.
        imgs_paths: a list of paths to images.
        layers_names: a list of layers names to calculate RDMs for.
    
    Returns:
        A dictionary of RDMs for each layer.
    '''
    # Set up input:
    # TODO: Set maximum batch size
    all_imgs = [preprocess(Image.open(img_pth)).unsqueeze(0) for img_pth in imgs_paths]
    all_imgs = torch.concat(all_imgs).cuda()

    # Encode input:
    with torch.no_grad(), torch.cuda.amp.autocast():
        model_history = tl.log_forward_pass(model, all_imgs, layers_to_save=[l for l in layers_names.values()], vis_opt='none')

    model.cpu()
    del model

    # Get layers:
    layers = {layer_name: model_history[layer_name].tensor_contents for layer_name in layers_names}

    del model_history

    # Join all layers
    layers = {name: torch.concat(layers[name]) for name in layers}

    # Flatten each layer
    layers = {name: layers[name].reshape((layers[name].shape[0], -1)) for name in layers}

    # Calculate RDM
    # TODO: Make metric configurable
    layers = {name: F.pairwise_cosine_similarity(layers[name]).cpu().numpy() for name in layers}

    return layers
