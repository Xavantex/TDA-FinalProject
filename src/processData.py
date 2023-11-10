from numpy import ndarray, shape, zeros, argsort, arange, max, min, shape
from numpy.random import choice

from scipy.spatial.distance import cdist, pdist, squareform

## Center data w.r.t. a point
def center(data: ndarray, point: ndarray):
    assert shape(data)[1:] == (3,)

    urb = urb - point


## filter out data w.r.t. a radii
def radiiFilter(data: ndarray, radii: float):
    assert shape(data)[1:] == (3,)

    # Radii of 4 meter from center
    surb = data[:,0]**2 * data[:,1]**2 < radii
    return data[surb]


## randomly sample number of points
def xSamples(data: ndarray, numPoints: int):
    assert shape(data)[1:] == (3,)

    return data[choice(shape(data)[0], numPoints)]


## choose random point, sort by the closest points and choose the 100 closest points.
def sampleNclosePoints(data: ndarray, numPoints: int, closePoints: int):
    assert shape(data)[1:] == (3,)

    rurb = choice(shape(data)[0], numPoints)

    curb = zeros((numPoints, closePoints, 3))

    for idx, sample in enumerate(rurb):
        tmp = data[argsort(cdist(data, data[None, sample]).squeeze())]
        curb[idx] = tmp[:closePoints]


    return curb


# Divide data in z-dimension in equally parts given a min, max and num sections values
# If no min, max value given, assumes min, max of data as min, max
def sectionData(data: ndarray, sects: int, minimum: float=None, maximum: float=None):
    datashape = shape(data)

    assert datashape[1:] == (3,)

    if maximum is None:
        maximum = max(data[:,2])
    if minimum is None:
        minimum = min(data[:,2])

    dstep = maximum - minimum / float(sects)

    parts = arange(minimum, maximum, dstep)

    sdata = []
    pSize = parts.shape[0]
    for idx, part in enumerate(parts):
        if idx+1 == pSize:
            sdata.append(data[(data[:,2] >= part) & (data[:,2] <= maximum)])
        else:
            sdata.append(data[(data[:,2] >= part) & (data[:,2] < parts[idx+1])])

    return sdata