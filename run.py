import json
import argparse

from src.config import settings
from src.modules import run_training


def get_params(path_file):
    with open(path_file, 'r') as f:
        params = json.load(f)
    
    return params


def main(args):    

    if args.training:
        training_params = get_params(settings.PATH_TRAIN_PARAMS)
        run_training.main(training_params, settings.PATH_DATA, settings.PATH_DATASET, settings.PATH_TRAIN_PARAMS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para treinar o gerador")

    parser.add_argument('--training', action='store_true', help='Se verdadeiro, executa o treinamento')

    args = parser.parse_args()
    
    main(args)
