import argparse
import importlib
import json
import os


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experimentFolder', type=str, help="Folder name of the experiment you want to run")
    return parser


def load_config(fp):
    with open(fp, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.experimentFolder):
        raise ValueError

    mymodule = importlib.import_module('.train', ''.join(
        e for e in args.experimentFolder if e.isalnum()))

    config = load_config(os.path.join(os.getcwd(), args.experimentFolder, "config.json"))

    mymodule.run_experiment(fp=os.path.join(os.getcwd(), args.experimentFolder), **config)
