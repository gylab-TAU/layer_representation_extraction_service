from torch import nn
import numpy as np
from typing import Callable, Dict, Any, List
from PIL import Image
from torch import Tensor


class RDMCalculator(object):
    """Base class for RDM calculators."""

    def calc_rdm(self, model: nn.Module, preprocess: Callable[[Image.Image], Tensor], imgs_paths: List[str], layers_names: Dict[Any, str]) -> Dict[str, np.ndarray]:
        """
        Calculate RDMs for a given model and images.

        Args:
            model: the model used to get the representations.
            preprocess: a function that preprocesses an image to a tensor (model compatible).
            imgs_paths: a list of paths to images.
            layers_names: a list of layers names to calculate RDMs for.

        Returns:
            A dictionary of RDMs for each layer.
        """
        raise NotImplementedError('RDM calculator is a base class and should not be instantiated.')