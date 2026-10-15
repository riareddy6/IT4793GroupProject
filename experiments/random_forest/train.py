"""Independent random_forest training entry point."""
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from experiments.common.data_loader import load_config
from experiments.common.evaluation import run_classifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    args = parser.parse_args()
    config = load_config(args.config)
    model = RandomForestClassifier(**config['parameters'])
    run_classifier('random_forest', args.config, model, scale=False)

if __name__ == '__main__':
    main()
