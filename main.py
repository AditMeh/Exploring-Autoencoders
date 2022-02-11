import argparse
import importlib
import json
import os


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experimentFolder', type=str, help="Folder name of the experiment you want to run")
    parser.add_argument(
        '-v', "--visualize", help="if specified, the visualization script for the experiment will run", action="store_true")
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

    experiment_name = os.path.split(os.path.abspath(args.experimentFolder))[-1]

    if not args.visualize:
        mymodule = importlib.import_module(
            '.train', '.'.join(['Experiments', experiment_name]))

        config = load_config(os.path.join(
            os.getcwd(), args.experimentFolder, "config.json"))
        mymodule.run_experiment(fp=os.path.join(
            os.getcwd(), args.experimentFolder), **config)

    else:
        mymodule = importlib.import_module(
            '.visualize', '.'.join(['Experiments', experiment_name]))

        config = load_config(os.path.join(
            os.getcwd(), args.experimentFolder, "config.json"))

        mymodule.visualize(fp=os.path.join(os.getcwd(
        ), args.experimentFolder), architecture_params=config["architecture_params"])
