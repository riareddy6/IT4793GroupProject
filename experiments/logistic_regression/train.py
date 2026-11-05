"""Independent logistic_regression training entry point."""
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from experiments.common.data_loader import load_config
from experiments.common.evaluation import run_classifier

def main():
    """Train the configured logistic_regression classifier through the shared evaluation pipeline.

    Args:
        None: CLI --config defaults to config.json beside this module.

    Returns:
        None: Prints the generated artifact directory or report; command-line side effects described below.

    Processing:
        Parse configuration, construct the logistic_regression sklearn estimator with configured parameters and call run_classifier with its established scaling policy.

    Raises:
        SystemExit: For invalid CLI arguments. Data/training errors propagate. Writes a new run through run_classifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    args = parser.parse_args()
    config = load_config(args.config)
    model = LogisticRegression(**config['parameters'])
    run_classifier('logistic_regression', args.config, model, scale=True)

if __name__ == '__main__':
    main()
