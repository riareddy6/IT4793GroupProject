"""Independent decision_tree training entry point."""
import argparse
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from experiments.common.data_loader import load_config
from experiments.common.evaluation import run_classifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    args = parser.parse_args()
    config = load_config(args.config)
    model = DecisionTreeClassifier(**config['parameters'])
    run_classifier('decision_tree', args.config, model, scale=False)

if __name__ == '__main__':
    main()
