"""Exercise evaluation invariants and leakage-sensitive preprocessing boundaries."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from experiments.common.data_loader import PROJECT, load_config, load_data
from experiments.common.evaluation import metrics, select_threshold
from experiments.common.preprocessing import features, preprocessor
from experiments.common.result_saver import save_json
from experiments.common.split_manager import shared_split
from experiments.milestone3.build import read_json
from experiments.milestone3.documentation import source_files
import ast


class PipelineTests(unittest.TestCase):
    """Tests use repository fixtures and temporary outputs; no model tuning."""

    def test_metric_orientation_and_probability_scores(self):
        """Check inclusive thresholds and asymmetric TN/FP/FN/TP counts.

        Args:
            self (PipelineTests): unittest instance.
        Returns:
            None: Asserts metric and sklearn probability-score agreement.
        Processing:
            Score hand-labeled examples including a probability equal to cutoff.
        Raises:
            AssertionError: On incorrect counts or ranking metrics. No file writes.
        """
        y = [0, 0, 0, 1, 1, 1, 1]
        p = [0.1, 0.2, 0.8, 0.1, 0.2, 0.3, 0.5]
        result = metrics(y, p, 0.5)
        self.assertEqual(result['confusion_matrix'], [[2, 1], [3, 1]])
        self.assertAlmostEqual(result['precision'], 0.5)
        self.assertAlmostEqual(result['recall'], 0.25)
        self.assertAlmostEqual(result['f1'], 1/3)
        self.assertAlmostEqual(result['roc_auc'], roc_auc_score(y, p))
        self.assertAlmostEqual(result['average_precision'], average_precision_score(y, p))

    def test_invalid_metrics_and_one_class(self):
        """Reject malformed metric inputs and retain fixed one-class matrix shape.

        Args:
            self (PipelineTests): unittest instance.
        Returns:
            None: Invalid inputs raise; one-class ROC AUC is undefined (None).
        Processing:
            Exercise empty, unaligned, nonfinite, out-of-range and nonbinary cases.
        Raises:
            AssertionError: If validation or fixed-label matrix behavior regresses.
        """
        for y, p, threshold in (([], [], .5), ([0, 1], [.5], .5), ([0, 2], [.1, .2], .5),
                                 ([0, 1], [np.nan, .2], .5), ([0, 1], [.1, 1.1], .5), ([0, 1], [.1, .2], 2)):
            with self.assertRaises(ValueError):
                metrics(y, p, threshold)
        result = metrics([1, 1], [.8, .9])
        self.assertIsNone(result['roc_auc'])
        self.assertEqual(result['confusion_matrix'], [[0, 0], [0, 2]])

    def test_threshold_ties_and_training_only_interface(self):
        """Verify threshold tie-breaking without accepting test predictions.

        Args:
            self (PipelineTests): unittest instance.
        Returns:
            None: Asserts F1, precision and proximity ordering.
        Processing:
            Use conflicting development scores to test both tie-break priorities.
        Raises:
            AssertionError: If selected thresholds differ or invalid rows pass.
        """
        table = pd.DataFrame({'threshold': [.2, .6, .5], 'f1': [.7, .7, .69], 'precision': [.6, .6, .9]})
        self.assertEqual(select_threshold(table), .6)
        table.loc[0, 'precision'] = .8
        self.assertEqual(select_threshold(table), .2)
        with self.assertRaises(ValueError):
            select_threshold(table.iloc[:0])

    def test_fold_local_imputation_and_unknown_categories(self):
        """Ensure held-out values do not alter fitted numeric imputation statistics.

        Args:
            self (PipelineTests): unittest instance.
        Returns:
            None: Asserts median, unseen-category handling and finite transforms.
        Processing:
            Fit on small training data, transform extreme/unknown validation data.
        Raises:
            AssertionError: On leaked statistics or broken unknown-category support.
        """
        train = pd.DataFrame({'Income': [10., 20., np.nan], 'Education': ['A', 'B', 'A']})
        heldout = pd.DataFrame({'Income': [np.nan, 999999.], 'Education': ['C', 'A']})
        prep = preprocessor(train, scale=True).fit(train)
        transformed = prep.transform(heldout)
        self.assertEqual(prep.named_transformers_['numeric'].named_steps['imputer'].statistics_[0], 15.)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape, (2, 3))
        self.assertTrue(np.all(transformed[0, 1:] == 0))

    def test_data_features_and_shared_folds(self):
        """Verify project dimensions, restored Income and disjoint persisted folds.

        Args:
            self (PipelineTests): unittest instance.
        Returns:
            None: Asserts established data counts, features and fold coverage.
        Processing:
            Load real data/configuration and compare absolute versus relative indices.
        Raises:
            AssertionError: On changed data/splits; loader failures propagate.
        """
        c = load_config(PROJECT/'experiments/logistic_regression/config.json')
        data, audit = load_data(c)
        self.assertEqual(data.shape, (2237, 33))
        self.assertEqual(data.Response.value_counts().to_dict(), {0: 1903, 1: 334})
        self.assertEqual(audit['restored_missing_income'], 25)
        self.assertEqual(data.Income.isna().sum(), 25)
        x = features(data, c)
        self.assertEqual(x.shape, (2237, 28))
        self.assertFalse({'ID', 'Response', 'cluster', 'Year_Birth', 'Dt_Customer'} & set(x))
        self.assertTrue((x.Customer_Age == 2014-data.Year_Birth).all())
        train, test, folds, _ = shared_split(data, c)
        self.assertEqual((len(train), len(test)), (1789, 448))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(sorted(np.concatenate([b for _, b in folds])), list(range(len(train))))
        for a, b in folds:
            self.assertFalse(set(a) & set(b))
            self.assertFalse(set(train[b]) & set(test))

    def test_config_validation_and_path_resolution(self):
        """Require historical split settings and repository-relative data resolution.

        Args:
            self (PipelineTests): unittest instance.
        Returns:
            None: Asserts path resolution and rejection of incompatible settings.
        Processing:
            Modify only a temporary configuration, leaving source configuration intact.
        Raises:
            AssertionError: If an incompatible seed is accepted. Writes temporary JSON.
        """
        source = PROJECT/'experiments/logistic_regression/config.json'
        c = load_config(source)
        self.assertTrue(Path(c['dataset']).is_absolute())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            invalid = read_json(source); invalid['random_state'] = 7
            save_json(path, invalid)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_strict_json_serialization(self):
        """Check UTF-8 round trips and rejection of nonfinite JSON payloads.

        Args:
            self (PipelineTests): unittest instance.
        Returns:
            None: Asserts exact payload recovery and invalid-number failure.
        Processing:
            Write/read a temporary JSON and attempt NaN serialization.
        Raises:
            AssertionError: For serialization regressions. Temporary files are cleaned.
        """
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'out.json'
            value = {'text': '验证', 'matrix': [[1, 2], [3, 4]]}
            save_json(path, value)
            self.assertEqual(read_json(path), value)
            with self.assertRaises(ValueError):
                save_json(path, {'invalid': float('nan')})

    def test_all_functions_documented_and_parseable(self):
        """Require complete documentation sections for every owned Python function.

        Args:
            self (PipelineTests): unittest instance.
        Returns:
            None: Asserts parsability and section coverage, including nested tests.
        Processing:
            Walk ASTs without importing side-effectful historical scripts.
        Raises:
            AssertionError: If any function lacks its required documentation.
        """
        for path in source_files():
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node) or ''
                    for section in ('Args:', 'Returns:', 'Processing:', 'Raises:'):
                        self.assertIn(section, doc, f'{path}:{node.name}')

    def test_generated_bundle_matches_source_scores(self):
        """Check dashboard/export metrics and profiles against saved source evidence.

        Args:
            self (PipelineTests): unittest instance.
        Returns:
            None: Asserts all four rows, selected thresholds and response counts.
        Processing:
            Read the generated bundle and recompute from source prediction files.
        Raises:
            AssertionError: If generated data drifts from its pinned sources.
        """
        bundle = read_json(PROJECT/'docs/milestone3/dashboard_data.json')
        self.assertEqual(len(bundle['comparison']), 4)
        for row in bundle['comparison']:
            run = PROJECT/bundle['sources'][row['model']]
            pred = pd.read_csv(run/'test_predictions.csv')
            actual = metrics(pred.true_label, pred.predicted_probability, row['threshold'])
            for key in ('accuracy','precision','recall','f1','roc_auc','average_precision'):
                self.assertAlmostEqual(row['test_'+key], actual[key], places=12)
        self.assertEqual(sum(p['size'] for p in bundle['clusters']['profiles']), 2237)
        self.assertEqual(sum(p['responders'] for p in bundle['clusters']['profiles']), 334)


if __name__ == '__main__':
    unittest.main()
