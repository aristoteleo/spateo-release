import numpy as np

def procrustes(X, Y, scaling=True, reflection='best'):
    """ This function will need to be rewritten just with scipy.spatial.procrustes and
    scipy.linalg.orthogonal_procrustes later.

    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Parameters
    ----------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Returns
    -------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = np.linalg.norm(X0, 'fro')**2 #(X0**2.).sum()
    ssY = np.linalg.norm(Y0, 'fro')**2 #(Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b*np.dot(muY, T)

    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


def AffineTrans(x, y, centroid_x, centroid_y, theta):
    """Translate the x/y coordinates of data points by the translating the centroid to the origin. Then data will be
    rotated with angle theta.

    Parameters
    ----------
        x: `np.array`
            x coordinates for the data points (bins). 1D np.array.
        y: `np.array`
            y coordinates for the data points (bins). 1D np.array.
        centroid_x: `float`
            x coordinates for the centroid of data points (bins).
        centroid_y: `np.array`
            y coordinates for the centroid of data points (bins).
        theta: `float`
            the angle of rotation. Unit is is in `np.pi` (so 90 degree is `np.pi / 2` and value is defined in the
            clockwise direction.

    Returns
    -------
        T_t: `np.array`
            The translation matrix used in affine transformation.
        T_r: `np.array`
            The rotation matrix used in affine transformation.
        trans_xy_coord: `np.array`
            The matrix that stores the translated and rotated coordinates.
    """

    trans_xy_coord = np.zeros((len(x), 2))

    T_t, T_r = np.zeros((3, 3)), np.zeros((3, 3))
    np.fill_diagonal(T_t, 1)
    np.fill_diagonal(T_r, 1)

    T_t[0, 2], T_t[1, 2] = -centroid_x[0], -centroid_y[0]

    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    T_r[0, 0], T_r[0, 1] = cos_theta, sin_theta
    T_r[1, 0], T_r[1, 1] = - sin_theta, cos_theta

    for cur_x, cur_y, cur_ind in zip(x, y, np.arange(len(x))):
        data = np.array([cur_x, cur_y, 1])
        res = T_t @ data
        res = T_r @ res
        trans_xy_coord[cur_ind, :] = res[:2]

    return T_t, T_r, trans_xy_coord
