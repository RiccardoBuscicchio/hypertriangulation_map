import numpy as np
import warnings

def hypertriangulate(x, bounds=(0, 1)):
    """
    Transform a vector of numbers from a cube to a hypertriangle.
    The hypercube is the space the samplers work in, and the hypertriangle is
    the physical space where the components of x are ordered such that
    x0 < x1 < ... < xn. The (unit) transformation is defined by:

    .. math::
        X_j = 1 - \\prod_{i=0}^{j} (1 - x_i)^{1/(K-i)}

    Parameters
    ----------

    x: array
        The hypercube parameter values

    bounds: tuple
        Lower and upper bounds of parameter space. Default is to transform
        between the unit hypercube and unit hypertriangle.

    Returns
    -------

    X: array
        The hypertriangle parameter values
    """

    # transform to the unit hypercube
    unit_x = (np.array(x) - bounds[0]) / (bounds[1] - bounds[0])

    # hypertriangle transformation
    with warnings.catch_warnings():
        # this specific warning is raised when unit_x goes outside [0, 1]
        warnings.filterwarnings('error', 'invalid value encountered in power')
        try:
            K = np.size(unit_x)
            index = np.arange(K)
            inner_term = np.power(1 - unit_x, 1/(K - index))
            unit_X = 1 - np.cumprod(inner_term)
        except RuntimeWarning:
            raise ValueError('Values outside bounds passed to hypertriangulate')

    # re-apply orginal scaling, offset
    X = bounds[0] + unit_X * (bounds[1] - bounds[0])

    return X
    
# if __name__ == "__main__":
    # add a two-dimensional example (in the vein of figure 1)
