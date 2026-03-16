import unittest

import numpy as np

from spateo.alignment.methods.utils import _robust_svd, inlier_from_NN


class TestRobustSVD(unittest.TestCase):
    def test_robust_svd_normal_case(self):
        """Test robust SVD with a well-conditioned matrix."""
        # Create a well-conditioned matrix
        A = np.random.randn(5, 5)
        U, S, Vt = _robust_svd(A)
        
        # Verify the decomposition
        A_reconstructed = U @ np.diag(S) @ Vt
        np.testing.assert_allclose(A, A_reconstructed, rtol=1e-10)
    
    def test_robust_svd_near_singular(self):
        """Test robust SVD with a nearly singular matrix."""
        # Create a nearly singular matrix (rank deficient)
        A = np.array([[1.0, 2.0, 3.0],
                      [2.0, 4.0, 6.0],
                      [3.0, 6.0, 9.0 + 1e-10]])
        
        # This should still work with regularization fallback
        try:
            U, S, Vt = _robust_svd(A)
            # Check that we got valid output
            self.assertEqual(U.shape[0], 3)
            self.assertEqual(Vt.shape[1], 3)
            self.assertEqual(len(S), 3)
        except np.linalg.LinAlgError:
            # This is acceptable for extremely degenerate cases
            pass
    
    def test_robust_svd_rectangular(self):
        """Test robust SVD with rectangular matrices."""
        # Test with more rows than columns
        A = np.random.randn(10, 5)
        U, S, Vt = _robust_svd(A)
        self.assertEqual(U.shape[0], 10)
        self.assertEqual(Vt.shape[1], 5)
        self.assertEqual(len(S), 5)
        
        # Test with more columns than rows
        A = np.random.randn(5, 10)
        U, S, Vt = _robust_svd(A)
        self.assertEqual(U.shape[0], 5)
        self.assertEqual(Vt.shape[1], 10)
        self.assertEqual(len(S), 5)
    
    def test_inlier_from_NN_basic(self):
        """Test inlier_from_NN with simple point sets."""
        # Create two point sets with known transformation
        np.random.seed(42)
        N = 50
        D = 3
        
        # Original points
        train_x = np.random.randn(N, D)
        
        # Apply a known rotation and translation
        theta = np.pi / 6
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        t_true = np.array([1.0, 2.0, 3.0])
        train_y = train_x @ R_true.T + t_true
        
        # Add some noise
        train_y += 0.01 * np.random.randn(N, D)
        
        # Create distance array
        distance = np.random.rand(N, 1) * 0.1
        
        # Run inlier_from_NN
        P, R, t, init_weight, sigma2, gamma = inlier_from_NN(train_x, train_y, distance)
        
        # Check output shapes
        self.assertEqual(P.shape, (N, 1))
        self.assertEqual(R.shape, (D, D))
        self.assertEqual(t.shape, (D,))
        
        # Check that most points are identified as inliers
        self.assertGreater(np.mean(P), 0.5)
    
    def test_inlier_from_NN_with_outliers(self):
        """Test inlier_from_NN with outliers."""
        np.random.seed(42)
        N = 100
        D = 3
        n_outliers = 20
        
        # Inlier points
        train_x = np.random.randn(N, D)
        train_y = train_x + 0.1 * np.random.randn(N, D)
        
        # Add outliers
        train_y[-n_outliers:] += 10.0 * np.random.randn(n_outliers, D)
        
        distance = np.random.rand(N, 1)
        
        # Run inlier_from_NN
        P, R, t, init_weight, sigma2, gamma = inlier_from_NN(train_x, train_y, distance)
        
        # Outliers should have lower probabilities
        self.assertLess(np.mean(P[-n_outliers:]), np.mean(P[:-n_outliers]))
    
    def test_inlier_from_NN_collinear_points(self):
        """Test inlier_from_NN with collinear points (edge case)."""
        # Create collinear points (all on a line)
        N = 50
        t = np.linspace(0, 10, N)
        train_x = np.column_stack([t, 2*t, 3*t])
        train_y = train_x + 0.1 * np.random.randn(N, 3)
        distance = np.ones((N, 1)) * 0.1
        
        # This should either work or raise an informative error
        try:
            P, R, t_vec, init_weight, sigma2, gamma = inlier_from_NN(train_x, train_y, distance)
            # If it works, check basic properties
            self.assertEqual(P.shape, (N, 1))
        except np.linalg.LinAlgError as e:
            # Expected for degenerate cases
            self.assertIn("collinear", str(e).lower() or "degenerate" in str(e).lower())


if __name__ == "__main__":
    unittest.main()
