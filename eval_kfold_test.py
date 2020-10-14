import argparse
from pathlib import Path
from tqdm import trange, tqdm
import _pickle as pickle
import numpy as np

networks = {
    'cnn': 'cnn_cnn',
    'cnn-resnet': 'cnn_resnet',
    'cnn-cbam': 'cnn_cbam'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=list(networks.keys()), required=True)
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    prefix = networks[args.network]

    model_dir = Path('model')
    agg_df = None
    for fold_index in trange(4):
        metric_dir = model_dir / f'{args.task}_{prefix}_fold{fold_index}'
        with open(metric_dir / 'result_metric.pickle', 'rb') as fp:
            metrics = pickle.load(fp)
            if agg_df is None:
                agg_df = metrics['test']['Agg']
            else:
                agg_df += metrics['test']['Agg']

    metric_dir = Path('metric') / f'{args.task}_{prefix}'
    metric_dir.mkdir(exist_ok=True, parents=True)
    agg_df['Accuracy'] = np.array([
        (agg_df.iloc[i, i] / agg_df.iloc[:, i].sum())
        for i in range(len(agg_df))
    ])
    agg_df.to_html(metric_dir / 'ConfusionMatrix.html')


if __name__ == "__main__":
    main()
