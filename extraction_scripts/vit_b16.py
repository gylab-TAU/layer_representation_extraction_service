from transformers import AutoImageProcessor, ViTForImageClassification, ViTFeatureExtractor, ViTModel
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
    parser.add_argument('--model_arch', type=str, help='The model architecture to use', default="google/vit-base-patch16-224")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_params()
    # ('google/vit-base-patch16-224', 'ViT-B16 - ImageNet'), ('google/vit-base-patch32-384', 'ViT-B32 - ImageNet'), ('google/vit-large-patch16-224', 'ViT-L16 - ImageNet'), 
    
    for arch, dir in tqdm([('google/vit-huge-patch14-224-in21k', 'ViT-H14 - ImageNet21k')]):
        (args.output_dir / dir).mkdir(parents=True, exist_ok=True)
        layer_names = set()
        image_processor = AutoImageProcessor.from_pretrained(arch)

        if arch == 'google/vit-huge-patch14-224-in21k':
            model = ViTModel.from_pretrained(arch)
            image_processor = ViTFeatureExtractor.from_pretrained(arch)
        else:
            model = ViTForImageClassification.from_pretrained(arch)
            image_processor = AutoImageProcessor.from_pretrained(arch)
        
        imgs_names = []
        for img in tqdm(args.input_dir.glob('*.png')):
            (args.output_dir / dir / img.stem).mkdir(parents=True, exist_ok=True)
            imgs_names.append(img.stem)
            img = Image.open(img)
            
            inputs = image_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                for i, layer in enumerate(outputs.hidden_states):
                    layer_names.add(i)
                    torch.save(outputs.hidden_states[i].flatten(), args.output_dir / dir / imgs_names[-1] / f'{i}.pkl')
                layer_names.add('Output')
                if arch == 'google/vit-huge-patch14-224-in21k':
                    torch.save(outputs.last_hidden_state.flatten(), args.output_dir / dir / imgs_names[-1] / f'Output.pkl')
                    layer_names.add('OutputCls')
                    torch.save(outputs.last_hidden_state[:, 0, :].flatten(), args.output_dir / dir / imgs_names[-1] / f'OutputCls.pkl')

                else:
                    torch.save(outputs.hidden_state[-1][:, 0, :].flatten(), args.output_dir / dir / imgs_names[-1] / f'Output.pkl')
                    layer_names.add('Logits')
                    torch.save(outputs.logits.squeeze(0), args.output_dir / dir / imgs_names[-1] / f'Logits.pkl')
        

        # matrices = {i: torch.stack(layers[i]) for i in layers}
        # rdms = {i: 1 - tmf.pairwise_cosine_similarity(matrices[i], zero_diagonal=False) for i in matrices}

        # (args.output_dir / dir).mkdir(parents=True, exist_ok=True)
        # for i in rdms:
        #     pd.DataFrame(rdms[i], index=imgs_names, columns=imgs_names).to_csv(args.output_dir / dir / f'{i}.csv')
        del image_processor
        del model
        
        for i in tqdm(layer_names):
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
            