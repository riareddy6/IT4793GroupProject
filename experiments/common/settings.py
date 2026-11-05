"""Established experiment constants shared by training and reporting."""
MODELS = ('logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting')
LABELS = dict(zip(MODELS, ('Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting')))
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.5
THRESHOLDS = tuple(round(0.10 + i * 0.05, 2) for i in range(17))
PERMUTATION_REPEATS = 5
