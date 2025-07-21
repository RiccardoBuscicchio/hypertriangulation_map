import numpy as np
import pytest
from hypertransform.hypertransform import hypertriangulate, hypercubify

def test_vectorized_vs_loop_equivalence():
    num_dim, num_points = 2, 1000
    bounds = (-1, 1)

    x = np.random.uniform(*bounds, size=(num_points, num_dim))

    # Loop-based transformation
    x_transformed = np.zeros_like(x)
    for i in range(len(x)):
        x_transformed[i] = hypertriangulate(x[i], bounds=bounds)

    # Vectorized transformation
    x_transformed_vectorized = hypertriangulate(x, bounds=bounds)

    assert np.allclose(x_transformed, x_transformed_vectorized)

def test_ordering_of_transformed_points():
    num_dim, num_points = 2, 1000
    bounds = (-1, 1)
    x = np.random.uniform(*bounds, size=(num_points, num_dim))
    x_transformed = hypertriangulate(x.T, bounds=bounds)
    
    assert np.all(np.diff(x_transformed, axis=1) >= 0), \
        "points in hypertriangle are not ordered correctly"

def test_inverse_transformation():
    num_dim, num_points = 2, 1000
    bounds = (-1, 1)
    x = np.random.uniform(*bounds, size=(num_points, num_dim))
    y = hypertriangulate(x, bounds=bounds)
    z = hypercubify(y, bounds=bounds)
    
    assert np.allclose(x, z)

def test_batch_vs_loop_equivalence():
    num_dim, num_points = 2, 1000
    bounds = (-1, 1)
    x = np.random.uniform(*bounds, size=(num_points//10, num_points//10, num_dim))

    y_vectorized = hypertriangulate(x, bounds=bounds)

    y_loop = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y_loop[i, j] = hypertriangulate(x[i, j], bounds=bounds)

    z = hypercubify(y_vectorized, bounds=bounds)

    assert np.allclose(y_loop, y_vectorized)
    assert np.allclose(x, z)
