"""Independent gradient_boosting training entry point."""
import argparse
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from experiments.common.data_loader import load_config
from experiments.common.evaluation import run_classifier

def main():
    """Train the configured gradient_boosting classifier through the shared evaluation pipeline.

    Args:
        None: CLI --config defaults to config.json beside this module.

    Returns:
        None: Prints the generated artifact directory or report; command-line side effects described below.

    Processing:
        Parse configuration, construct the gradient_boosting sklearn estimator with configured parameters and call run_classifier with its established scaling policy.

    Raises:
        SystemExit: For invalid CLI arguments. Data/training errors propagate. Writes a new run through run_classifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    args = parser.parse_args()
    config = load_config(args.config)
    model = GradientBoostingClassifier(**config['parameters'])
    run_classifier('gradient_boosting', args.config, model, scale=False)

if __name__ == '__main__':
    main()
