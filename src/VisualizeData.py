from typing import Union

from processData import sampleNclosePoints
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.spatial.distance import pdist, squareform
import stablerank.srank as sr
from numpy import ndarray

from ripser import ripser as rp
from persim import plot_diagrams


## visualize points RAW
def visRAW(nat: ndarray, urb: ndarray):
    assert nat.shape[0] == (3,)
    assert urb.shape[0] == (3,)
    ## Visualize all data, RAW
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(urb[:,0], urb[:,1], urb[:,2], marker='o')
    ax.scatter(nat[:,0], nat[:,1], nat[:,2], marker='^')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()


def visHN(data: ndarray, name: Union[str, float, int], axs: ndarray=None):

    datadist = sr.Distance(squareform(pdist(data, metric="euclidean")))
    #plot_diagrams(rp(datadist, maxdim=2, distance_matrix=True)['dgms'], show=True)

    if axs is None:
        fig, axs = plt.subplots(3, 1)

    ## Homo 0
    maxdim = 2
    bdat = datadist.get_bc(maxdim=maxdim)

    fdat = sr.bc_to_sr(bdat, degree="H0")

    pdat = fdat.plot(ax=axs[0], label=f'{name} H0')
    #axs[0].set_label(f'Urban {name} H0')


    ## Homo 1
    fdat = sr.bc_to_sr(bdat, degree="H1")

    pdat = fdat.plot(ax=axs[1], label=f'{name} H1')
    #axs[1].set_label(f'Urban {name} H1')


    ## Homo 2
    fdat = sr.bc_to_sr(bdat, degree="H2")

    pdat = fdat.plot(ax=axs[2], label=f'{name} H2')
    #axs[2].set_label()


## helper func
def __visRandNPointsHomoLoop(data: Union[ndarray, list], name: Union[str, float, int], axs: ndarray):

    #diagrams = rp(curbdist[0], maxdim=2, thresh= 4, distance_matrix=True)['dgms']
    for idx, cp in enumerate(data):
        visHN(cp, f'{name} {idx}', axs)


## Create homology from x sampled closest points
def visRandNPointsHomo(nat: Union[ndarray, list], urb: Union[ndarray, list]):
    ## choose random point, sort by the closest points and choose the 100 closest points.
    fig, axs = plt.subplots(3, 1)

    __visRandNPointsHomoLoop(urb, "Urban", axs)

    __visRandNPointsHomoLoop(nat, "Natural", axs)

    #fig.legend()
    for ax in axs:
        ax.legend()
    plt.show()


def visH0sr(data: ndarray):
    ## Create matrix distance
    DDAT = sr.Distance(squareform(pdist(data, metric="euclidean")))

    ## anoter stablerank method for H0 homo
    clustering_method = "complete"
    fnat = DDAT.get_h0sr(clustering_method=clustering_method)

    fnat.plot()