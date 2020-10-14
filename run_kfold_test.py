import papermill
import argparse
from pathlib import Path
from tqdm import trange, tqdm

networks = {
    'cnn': {
        'model_type': 'cnn',
        'file': 'CNN.ipynb',
    },
    'cnn-resnet': {
        'model_type': 'resnet',
        'file': 'CNN.ipynb',
    },
    'cnn-cbam': {
        'model_type': 'cbam',
        'file': 'CNN.ipynb',
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str,
                        choices=list(networks.keys()), required=True)
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    params = networks[args.network]
    file = params.pop('file')
    common_params = {
        'task': args.task,
    }
    common_params.update(params)

    name = Path(file).stem.lower()
    for fold_index in trange(4):
        params = {
            'fold_index': fold_index,
        }
        params.update(common_params)
        papermill.execute_notebook(
            file,
            str(Path('model') /
                f'{params["model_type"]}_{name}_fold{fold_index}.ipynb'),
            parameters=params
        )


if __name__ == "__main__":
    main()
