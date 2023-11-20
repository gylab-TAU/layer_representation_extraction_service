from models import get_clip_vit32_text_resources
from rdm_calculations import MemoryEfficientTextRDMCalculator
import torchmetrics.functional as F
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd


def get_params() -> dict:
    parser = ArgumentParser()
    parser.add_argument('--words_file', type=str, help='path to file with words to test')
    parser.add_argument('--words_col', type=str, help='Name of words column in words_file')
    parser.add_argument('--id_col', type=str, help='Name of id column in words_file')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--distance_metric', type=str, choices=['cos', 'l2'], help='distance metric to use for RDM calculation')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    return vars(parser.parse_args())


if __name__ == '__main__':
    params = get_params()
    if params['distance_metric'] == 'cos':
        distance_metric = F.pairwise_cosine_similarity
    elif params['distance_metric'] == 'l2':
        distance_metric = F.pairwise_euclidean_distance
    else:
        raise ValueError(f'Unknown distance metric: {params["distance_metric"]}')

    model, tokenizer, layers_names = get_clip_vit32_text_resources()
    model.eval()

    calc = MemoryEfficientTextRDMCalculator(distance_metric, params['batch_size'])

    df = pd.read_csv(params['words_file'])

    outputs = calc.calc_rdm(model, tokenizer, list(df[params['words_col']]), layers_names)

    for layer in outputs:
        Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(outputs[layer], index=df[params['id_col']], columns=df[params['id_col']]).to_csv(Path(params['output_dir']) / f'{layer}.csv')
        