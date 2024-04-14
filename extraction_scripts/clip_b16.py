from transformers import AutoProcessor, CLIPVisionModel
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


if __name__ == '__main__':
    args = get_params()
    #('openai/clip-vit-base-patch32', 'CLIP - ViTB-32'), ('openai/clip-vit-base-patch16', 'CLIP - ViTB-16'), ('openai/clip-vit-large-patch14', 'CLIP - ViTL-14'),
    for arch, dir in tqdm([('openai/clip-vit-large-patch14-336', 'CLIP - ViTL-14-336')]):
        image_processor = AutoProcessor.from_pretrained(arch)
        model = CLIPVisionModel.from_pretrained(arch)

        layers = {'Output': [], 'Pooler': []}
        imgs_names = []
        for img in tqdm(args.input_dir.glob('*.png')):
            imgs_names.append(img.stem)
            img = Image.open(img)
            
            inputs = image_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                for i, layer in enumerate(outputs.hidden_states):
                    if i not in layers:
                        layers[i] = []
                    layers[i].append(layer.flatten())
                layers['Output'].append(outputs.last_hidden_state[:, 0, :].flatten())
                layers['Pooler'].append(outputs.pooler_output.flatten())
        del image_processor
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
            
        
            


        