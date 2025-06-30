import unittest
import numpy as np
from src.utils.criterions import Gini, SquaredError


class TestCriterion(unittest.TestCase):
    def test_gini(self):
        pure_y = np.array([1, 1, 1, 1, 1, 1])
        impure_y = np.array([1, 1, 1, 0, 0, 0])
        empty_y = np.array([])

        gini = Gini()
        pure_gini = gini(pure_y)
        impure_gini = gini(impure_y)
        empty_gini = gini(empty_y)

        self.assertEqual(pure_gini, 0.0, msg=f"Gini must be 0.0, got {pure_gini}")
        self.assertEqual(impure_gini, 0.5, msg=f"Gini must be 0.5, got {impure_gini}")
        self.assertEqual(empty_gini, 1.0, msg=f"Gini must be 1.0, got {empty_gini}")

    def test_squared_error(self):
        pure_y = np.array([2.0, 2.0, 2.0])
        impure_y = np.array([2.5, 2.0, 1.5, 1.0, 3.0])
        empty_y = np.array([])

        se = SquaredError()
        pure_se = se(pure_y)
        impure_se = se(impure_y)
        empty_se = se(empty_y)

        self.assertEqual(pure_se, 0.0, msg=f"Squared error must be 0.0, got {pure_se}")
        self.assertEqual(impure_se, 0.5, msg=f"Squared error must be 0.5, got {impure_se}")
        self.assertEqual(empty_se, 0.0, msg=f"Squared error must be 0.0, got {empty_se}")


if __name__ == '__main__':
    unittest.main()
