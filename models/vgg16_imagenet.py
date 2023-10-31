import torchvision.transforms as t
from torchvision.models import vgg16, VGG16_Weights
from typing import Tuple, Callable, Dict, Any
from torch import Tensor, nn
from PIL import Image


def get_vgg16_imagenet_resources() -> Tuple[nn.Module, Callable[[Image.Image], Tensor], Dict[str, str]]:
    """
    Gets all resources for RDM extraction for ImageNet trained VGG16.

    Returns:
        A tuple of the model, preprocess function and layers names.
    """
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    # Set up input:
    all_imgs = [preprocess(Image.open(img_pth)).unsqueeze(0) for img_pth in imgs_paths]
    all_imgs = torch.concat(all_imgs)
    all_imgs = all_imgs.cuda()

    vgg16_layers = {'input': 0,#'input_1',
                    'Conv1': 'maxpool2d_1',#_5',
                    'Conv2': 'maxpool2d_2',#_10',
                    'Conv3': 'maxpool2d_3',#_17',
                    'Conv4': 'maxpool2d_4',#_24',
                    'Conv5': 'maxpool2d_5',#_31',
                    'FC6': 'relu_14',#_35',
                    'FC7': 'relu_15',#_38',
    }
    model.eval()

    return model, preprocess, vgg16_layers
