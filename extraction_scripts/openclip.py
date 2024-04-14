from ast import Not
from math import e
import open_clip
import torch
import pandas as pd
from PIL import Image
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torchmetrics.functional as tmf
from tqdm import tqdm
import pickle as pkl


def get_params() -> Namespace:
    parser = ArgumentParser(description='Extract RDMs from a model')
    parser.add_argument('input_dir', type=Path, help='The input directory containing the images')
    parser.add_argument('output_dir', type=Path, help='The output directory to save the RDMs')
    parser.add_argument('--model_arch', type=str, help='The model architecture to use', default="openai/clip-vit-base-patch16")
    return parser.parse_args()


def register_hooks(model, arch):
    layers_map_dict = {}
    if 'ViT' in arch:
        for i, block in enumerate(model.visual.transformer.resblocks):
            layers_map_dict[i] = block
            block.register_forward_hook(forward_hook)
        # model.transformer.resblocks.register_forward_hook(forward_hook)
        layers_map_dict['Output'] = model.visual.ln_post
        model.visual.ln_post.register_forward_hook(forward_hook)
    return layers_map_dict


def forward_hook(module, input, output):
    curr_layers_dict[module] = output.detach().flatten()


if __name__ == '__main__':    
    args = get_params()
    curr_layers_dict = {}
    for arch, pretrained in tqdm(open_clip.list_pretrained()):
        print(arch, pretrained)
        if not ((('ViT' in arch)) and (('laion' in pretrained) or ('datacomp' in pretrained))):# or ('openai' in pretrained))): # ('RN' in arch) or 
            continue
        dir = 'OpenCLIP-' + arch + '-' + pretrained
        if (args.output_dir / dir).exists():
            print('already done')
            continue
        model_tuple = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        if type(model_tuple) != tuple:
            continue
        else:
            if len(model_tuple) == 2:
                model, preprocess = model_tuple
            elif len(model_tuple) == 3:
                model, _, preprocess = model_tuple
                print(model_tuple[1], model_tuple[2])
            else:
                print(len(model_tuple))
                continue
        model.eval()
        layers_names = register_hooks(model, arch)

        layers = {}
        imgs_names = []
        for img in tqdm(args.input_dir.glob('*.png')):
            curr_layers_dict = {}
            imgs_names.append(img.stem)
            img = Image.open(img)
            
            inputs = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(inputs)

                for name in layers_names:
                    if name not in layers:
                        layers[name] = []
                    layers[name].append(curr_layers_dict[layers_names[name]])
                    del curr_layers_dict[layers_names[name]]
                del curr_layers_dict
                
        del preprocess
        del model
        (args.output_dir / dir).mkdir(parents=True, exist_ok=True)
        
        keys = list(layers.keys())
        
        for i in tqdm(keys):
            torch.save(layers[i], args.output_dir / dir / f'{i}.pkl')
            del layers[i]
        del layers

        for i in tqdm(keys):
            layer = torch.load(args.output_dir / dir / f'{i}.pkl')
            matrix = torch.stack(layer)
            del layer
            rdm = 1 - tmf.pairwise_cosine_similarity(matrix, zero_diagonal=False)
            del matrix
            pd.DataFrame(rdm.numpy(), index=imgs_names, columns=imgs_names).to_csv(args.output_dir / dir / f'{i}.csv')
            del rdm
            
        
            


        