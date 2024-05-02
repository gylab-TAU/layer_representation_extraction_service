import open_clip
import torch
import pandas as pd
from PIL import Image
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torchmetrics.functional as tmf
from tqdm import tqdm


def get_params() -> Namespace:
    parser = ArgumentParser(description='Extract RDMs from a model')
    parser.add_argument('input_dir', type=Path, help='The input directory containing the images')
    parser.add_argument('output_dir', type=Path, help='The output directory to save the RDMs')
    parser.add_argument('--model_arch', type=str, help='The model architecture to use', default="openai/clip-vit-base-patch16")
    return parser.parse_args()


def register_hooks(model, arch):
    name_layers_map_dict = {}
    layers_name_map_dict = {}
    if 'ViT' in arch:
        for i, block in enumerate(model.visual.transformer.resblocks):
            name_layers_map_dict[i] = block
            layers_name_map_dict[block] = i
            block.register_forward_hook(forward_hook)
        # model.transformer.resblocks.register_forward_hook(forward_hook)
        name_layers_map_dict['Output'] = model.visual.ln_post
        layers_name_map_dict[model.visual.ln_post] = 'Output'
        model.visual.ln_post.register_forward_hook(forward_hook)

    if 'RN' in arch:
        name_layers_map_dict['avgpool (first)'] = model.visual.avgpool
        layers_name_map_dict[model.visual.avgpool] = 'avgpool (first)'
        model.visual.avgpool.register_forward_hook(forward_hook)
        for layer in [model.visual.layer1, model.visual.layer2, model.visual.layer3, model.visual.layer4]:
            for i, bottleneck in enumerate(layer):
                name_layers_map_dict[f'Bottleneck.{i}'] = bottleneck
                layers_name_map_dict[bottleneck] = f'Bottleneck.{i}'
                bottleneck.register_forward_hook(forward_hook)

        name_layers_map_dict['attnpool'] = model.visual.attnpool
        layers_name_map_dict[model.visual.attnpool] = 'attnpool'
        model.visual.attnpool.register_forward_hook(forward_hook)

        # name_layers_map_dict['Output'] = model.ln_final
        # layers_name_map_dict[model.ln_final] = 'Output'
        # model.ln_final.register_forward_hook(forward_hook)
        
    return name_layers_map_dict, layers_name_map_dict


def forward_hook(module, input, output):
    # curr_layers_dict[names_layers[module]] = output.detach().flatten()
    (args.output_dir / dir / img_pth.stem).mkdir(parents=True, exist_ok=True)
    torch.save(output.detach().flatten(), args.output_dir / dir / img_pth.stem / f'{names_layers[module]}.pkl')


if __name__ == '__main__':    
    args = get_params()
    # curr_layers_dict = {}
    models = open_clip.list_pretrained()
    models = [('ViT-B-16', 'laion2B_s34B_b88K'), ('ViT-L-14', 'laion2B_s32B_b82K'), ('ViT-H-14','laion2B_s32B_b79K'), ('ViT-bigG-14','laion2B_39B_b160k')]
    for arch, pretrained in tqdm(models, desc='Iterating models...'):
        print(arch, pretrained)
        # if not ('ViT' in arch): #((('ViT' in arch)) and (('laion' in pretrained) or ('datacomp' in pretrained))):# or ('openai' in pretrained))): # ('RN' in arch) or 
        #     continue
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
        layers_names, names_layers = register_hooks(model, arch)

        layers = {}
        imgs_names = []
        for img_pth in tqdm(args.input_dir.glob('*.png'), desc='Extracting representations...'):
            # curr_layers_dict = {}
            imgs_names.append(img_pth.stem)
            img = Image.open(img_pth)
            
            inputs = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                model(inputs)
                
        del preprocess
        del model
        (args.output_dir / dir).mkdir(parents=True, exist_ok=True)
        
        for i in tqdm(layers_names, 'Calculating RDMs...'):
            reps = []
            for img in imgs_names:
                reps.append(torch.load(args.output_dir / dir / img / f'{i}.pkl'))
                # delete the file:
                (args.output_dir / dir / img / f'{i}.pkl').unlink()
            matrix = torch.stack(reps)
            del reps
            rdm = 1 - tmf.pairwise_cosine_similarity(matrix, zero_diagonal=False)
            del matrix
            pd.DataFrame(rdm.numpy(), index=imgs_names, columns=imgs_names).to_csv(args.output_dir / dir / f'{i}.csv')
            del rdm
        
        # delete the directories
        for img in imgs_names:
            (args.output_dir / dir / img).rmdir()
        
            
        
            


        
