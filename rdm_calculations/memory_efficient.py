from torch import nn, Tensor
import torch
import torchlens as tl
import torchmetrics.functional as F
from typing import Callable, List, Dict, Any
from PIL import Image
import numpy as np
from tqdm import tqdm


def extract_batch_representations(model: nn.Module, preprocess: Callable[[Image.Image], Tensor], imgs_paths: List[str], layers_names: Dict[str, Any]) -> Dict[str, Tensor]:
    """
    Extract representations for a batch of images.

    Args:
        model: the model to extract representations for.
        preprocess: a function that preprocesses an image to a tensor.
        imgs_paths: a list of paths to images.
        layers_names: a list of layers names to extract representations for.

    Returns:
        A dictionary of flattened representations for each layer.
    """
    batch = [preprocess(Image.open(img_pth)).unsqueeze(0) for img_pth in imgs_paths]
    batch = torch.concat(batch).cuda()

    with torch.no_grad(), torch.cuda.amp.autocast():
        model_history_a = tl.log_forward_pass(model, batch, layers_to_save=[l for l in layers_names.values()], vis_opt='none')

    # Get layers:
    layers = {layer_name: model_history_a[layer_name].tensor_contents for layer_name in layers_names}
    del model_history_a
    layers = {name: layers[name].reshape((layers[name].shape[0], -1)) for name in layers}

    return layers



def balanced_RDMs(model: nn.Module, preprocess: Callable[[Image.Image], Tensor], imgs_paths: List[str], layers_names: Dict[str, str], batch_size: int = 1) -> Dict[str, np.ndarray]:
    '''
    Calculate RDMs for a given model and images.
    To save memory, the calculation is done in batches of images, where each batch is calculated separately.

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

    rdm_rows = {}

    for i in tqdm(range(0, len(imgs_paths), batch_size)):
        curr_rdm_row = []

        # Load the batch for the vertical axis of the RDM:
        layers_a = extract_batch_representations(model, preprocess, imgs_paths[i : i+batch_size], layers_names)

        for j in range(i+1, len(imgs_paths), batch_size):
            layers_b = extract_batch_representations(model, preprocess, imgs_paths[j : j+batch_size], layers_names)

            for name in layers_names:
                # TODO: make metric configurable
                curr_rdm_row[name].append(F.pairwise_cosine_similarity(layers_a[name], layers_b[name]).cpu().numpy())
        
        for name in layers_names:
            rdm_rows[name].append(np.concat(curr_rdm_row[name], axis=1))

    model.cpu()
    del model

    # Join all layers
    layers = {name: torch.concat(rdm_rows[name], axis=0) for name in layers}

    return layers
