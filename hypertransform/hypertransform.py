import numpy as np
import warnings


def hypertriangulate(x, bounds=(0, 1)):
    """
    Transform a vector of numbers from the hypercube to the hypertriangle.

    The hypercube is the space the samplers usually work in; the 
    components of x are in no particular order.
    
    The hypertriangle is the space where the components are sorted into
    ascenting order, y0 < y1 < ... < yn. 
    
    The (unit) transformation is defined by:

    .. math::
        y_j = 1 - \\prod_{i=0}^{j} (1 - x_i)^{1/(n-i)}

    Example application. If we are analysing a number num_dim of DWD 
    sources, all with identical priors. Then this function would be
    called on the array np.array([f_1, f_2, ..., f_num_sources]) with
    bounds=(f_min, f_max).

    Parameters
    ----------
    x: array
        The hypercube parameter values. 
        The components of x are in no particular order.
        Input shape = (num_dim,) or (num_points, num_dim).
        If input array is multi-dimensional, the function is vectorised 
        along all except the last axis.

    bounds: tuple
        Lower and upper bounds of parameter space. Default is to transform
        between the unit hypercube and unit hypertriangle with (0, 1).

    Returns
    -------
    y: array, shaped like x
        The hypertriangle parameter values
        The components of y are sorted in ascending order.
    """
    x = np.array(x)
    
    # transform to the unit hypercube
    unit_x = (x - bounds[0]) / (bounds[1] - bounds[0])

    # hypertriangle transformation
    with warnings.catch_warnings():
        # this specific warning is raised when unit_x goes outside [0, 1]
        warnings.filterwarnings('error', 'invalid value encountered in power')
        try:
            n = np.size(unit_x, axis=-1)
            index = np.arange(n)
            inner_term = np.power(1 - unit_x, 1/(n - index))
            unit_y = 1 - np.cumprod(inner_term, axis=-1)
        except RuntimeWarning:
            raise ValueError('Values outside bounds passed to hypertriangulate')

    # re-apply orginal scaling, offset
    y = bounds[0] + unit_y * (bounds[1] - bounds[0])

    return y


def hypercubify(y, bounds=(0, 1)):
    """
    Inverse of hypertriangulate. Maps from the hypertriangle to the hypercube.
    
    The (unit) transformation is defined by:

    .. math::
        x_i = 1 - [(1 - y_i)/(1 - y_{i-1})]^{n-i}

    Parameters
    ----------
    y: array
        The hypertriangle parameter values.
        The components of y are sorted in ascending order.
        Input shape = (num_dim,) or (num_points, num_dim).
        If input array is multi-dimensional, the function is vectorised 
        along all except the last axis.

    bounds: tuple
        Lower and upper bounds of parameter space. Default is to transform
        between the unit hypercube and unit hypertriangle with (0, 1).

    Returns
    -------
    x: array, shaped like x
        The hypercube parameter values. 
        The components of x are in no particular order.
    """
    y = np.array(y)

    # transform to the unit hypertriangle
    unit_y = (y - bounds[0]) / (bounds[1] - bounds[0])

    # inverse hypertriangle transformation
    n = np.size(unit_y, axis=-1)
    index = np.arange(n)
    unit_y_shifted = np.roll(unit_y, 1, axis=-1)
    unit_y_shifted[...,0] = 0
    unit_x = 1 - np.power((1 - unit_y) / (1 - unit_y_shifted), n - index)

    # re-apply orginal scaling, offset
    x = bounds[0] + unit_x * (bounds[1] - bounds[0])

    return x
    
    
if __name__=='__main__':

    import matplotlib.pyplot as plt

    num_dim, num_points = 2, 1000
    bounds = (-1, 1)

    # points randomly distributed in the unit square
    x = np.random.uniform(*bounds, size=(num_points, num_dim))

    # transformed points in the hypertriangle
    x_transformed = np.zeros_like(x)
    for i in range(len(x)):
        x_transformed[i] = hypertriangulate(x[i], bounds=bounds)

    # test the vectorised version of the function
    x_transformed_vectorised = hypertriangulate(x, bounds=bounds)
    assert np.allclose(x_transformed, x_transformed_vectorised)

    # check that ordering is correct
    assert all(np.diff(x_transformed)>=0.), "points in hypertriangle are not ordered correctly"
    
    # transform points back to the hypercube
    z = np.zeros_like(x)
    for i in range(len(x)):
        z[i] = hypercubify(x_transformed[i], bounds=bounds)

    # check that the inverse transformation worked
    assert np.allclose(x, z)

    # test the vectorised version of the inverse function
    z_vectorised = hypercubify(x_transformed_vectorised, bounds=bounds)
    assert np.allclose(z_vectorised, z)
    
    # plot the points
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 5))
    ax0.scatter(x[:, 0], x[:, 1], s=10, color='r')
    ax1.scatter(x_transformed[:, 0], x_transformed[:, 1], s=10, color='b')
    for ax, title in zip([ax0, ax1],['hypercube','hypertriangle']):
        ax.set_xlim(*bounds)
        ax.set_ylim(*bounds)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_aspect('equal')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # one more test of the vectorised functions
    x = np.random.uniform(*bounds, size=(num_points//10, num_points//10, num_dim))
    y = hypertriangulate(x, bounds=bounds)
    y_ = np.zeros_like(x)
    for i in range(num_points//10):
        for j in range(num_points//10):
            y_[i,j] = hypertriangulate(x[i,j], bounds=bounds)
    z = hypercubify(y, bounds=bounds)
    assert np.allclose(x, z)
    assert np.allclose(y_, y)

    # tests passed
    print('Passed!')

