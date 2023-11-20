from torch import nn, Tensor
import numpy as np
import torch
import torchlens as tl
from typing import Callable, List, Dict, Any
from PIL import Image
import numpy as np
from tqdm import tqdm
from .rdm_calculator import RDMCalculator
import itertools


class MemoryEfficientTextRDMCalculator:
    """
    A memory efficient RDM calculator
    Loads the images in batches and calculates the RDMs for each batch separately.
    
    Memory complexity: O(batch_size * number_of_layers * number_of_images)
    Time complexity: O(number_of_layers * number_of_images^2 / batch_size)
    """
    def __init__(self, pairwise_similarity_metric: Callable[[Tensor, Tensor], Tensor], batch_size: int = 1):
        """
        Args:
            batch_size: the maximum number of images to load at once.
        """
        assert batch_size > 0, 'batch_size must be positive.'
        self.batch_size = batch_size
        self.pairwise_similarity_metric = pairwise_similarity_metric
        self.specific_layers = True
    
    
    def _extract_batch_representations(self, model: nn.Module, preprocess: Callable[[List[str]], Tensor], texts: List[str], layers_names: Dict[str, Any]) -> Dict[str, Tensor]:
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
        batch = preprocess(texts)
        if torch.cuda.is_available():
            batch = batch.cuda()
            model = model.cuda()
        else:
            model = model.cpu()
        

        with torch.no_grad():#, torch.cuda.amp.autocast():
            layers_string_title = [l for l in layers_names.values()]
            if self.specific_layers:
                try:
                    model_history_a = tl.log_forward_pass(model, batch, layers_to_save=layers_string_title, vis_opt='none')
                    
                except ValueError as ex:
                    print('Failed to calculate RDMs for specific layers. Falling back to all layers.')
                    self.specific_layers = False
                    model_history_a = tl.log_forward_pass(model, batch, layers_to_save='all', vis_opt='none')
                    model_history_a = {name: model_history_a[name] for name in layers_string_title}
            else:
                model_history_a = tl.log_forward_pass(model, batch, layers_to_save='all', vis_opt='none')
            
            layers = {name: model_history_a[name] for name in layers_string_title}
            del model_history_a

        # Get layers:
        layers = {name: layers[layers_names[name]].tensor_contents for name in layers_names}
        
        assert len(layers) == 0 or len(layers[next(iter(layers))]) == len(batch), 'The number of layers and the number of images are not equal.'

        # Flatten layers:
        flat_layers = {}
        for name in layers:
            # In case of transformers, we set the first dimension to batch size, and the second to the tokens.
            if 'transformer' in name:
                # reorder dimensions to [batch, tokens, features]
                layers[name] = layers[name].permute(1, 0, 2)
            flat_layers[name] = layers[name].reshape((layers[name].shape[0], -1))

        return flat_layers


    def calc_rdm(self, model: nn.Module, preprocess: Callable[[List[str]], Tensor], texts: List[str], layers_names: Dict[str, Any]) -> Dict[str, np.ndarray]:
        '''
        Calculate RDMs for a given model and images.
        To save memory, the calculation is done in batches of images, where each batch is calculated separately.

        Args:
            model: the model to calculate RDMs for.
            preprocess: a function that preprocesses an image to a tensor.
            texts: list of strings to encode.
            layers_names: a list of layers names to calculate RDMs for.
        
        Returns:
            A dictionary of RDMs for each layer.
        '''
        

        # Print the estimated complexity:
        n_batches = int(np.ceil(len(texts) / self.batch_size))

        # Set up input:
        block_matrices = {
            name: [
                [
                    None for block_col in range(n_batches) # A block for each batch
                ] for block_row in range(n_batches) # A block row for each batch
            ] for name in layers_names # A block matrix for each layer
        }

        print(f'n_batches = {len(texts)} / {self.batch_size} = {n_batches}')
        print(f'n_iters = {n_batches} + {n_batches} * ({n_batches} - 1) / 2 = {n_batches + n_batches*(n_batches-1)/2}')


        # Set up the pairs of batches:
        batches_start_indices = itertools.combinations_with_replacement(range(0, len(texts), self.batch_size), 2)
        batches_start_indices = list(batches_start_indices)

        prior_i = None
        pbar = tqdm(batches_start_indices)
        for i, j in pbar:
            # Get the block indices:
            block_row_idx = i // self.batch_size
            block_col_idx = j // self.batch_size
            pbar.set_description(f'Calculating block [{block_row_idx}, {block_col_idx}]')
            
            # Load the batch for the vertical axis of the RDM:
            if prior_i != i:
                layers_a = self._extract_batch_representations(model, preprocess, texts[i : i+self.batch_size], layers_names)
                prior_i = i
            
            # Load the batch for the horizontal axis of the RDM:
            if j == i:
                layers_b = layers_a
            else:
                layers_b = self._extract_batch_representations(model, preprocess, texts[j : j+self.batch_size], layers_names)

            for name in layers_names:
                distances = self.pairwise_similarity_metric(layers_a[name], layers_b[name]).cpu()
                
                # Set the symmetrically opposite blocks (RDMs are symmetric):
                block_matrices[name][block_row_idx][block_col_idx] = distances
                block_matrices[name][block_col_idx][block_row_idx] = distances.T

        model.cpu()
        del model
        
        # join all rows:
        rows = {name: [torch.concat(block_matrices[name][row_idx], axis=1) for row_idx in range(n_batches)] for name in block_matrices}
        # Join all layers
        layers = {name: torch.concat(rows[name], axis=0) for name in rows}

        return layers
