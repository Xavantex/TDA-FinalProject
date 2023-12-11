from typing import Union

from processData import sampleNclosePoints
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.spatial.distance import pdist, squareform
import stablerank.srank as sr
from numpy import ndarray, array, concatenate, empty, zeros

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

def _data2sr(data: ndarray):

    datadist = sr.Distance(squareform(pdist(data, metric="euclidean")))
    #plot_diagrams(rp(datadist, maxdim=2, distance_matrix=True)['dgms'], show=True)

    #if axs is None:
    #    fig, axs = plt.subplots(3, 1)

    ## Homo 0
    maxdim = 2
    bdat = datadist.get_bc(maxdim=maxdim)
    return sr.bc_to_sr(bdat, degree="H0").content, sr.bc_to_sr(bdat, degree="H1").content, sr.bc_to_sr(bdat, degree="H2").content

    #bdat["H1"].plot()

    #fdat = sr.bc_to_sr(bdat, degree="H0")

    #fdat.plot(ax=axs[0], label=f'{name} H0')
    #axs[0].set_label(f'Urban {name} H0')


    ## Homo 1
    #bdat["H1"].plot(ax=axs[1])
    #fdat = sr.bc_to_sr(bdat, degree="H1")

    #fdat.plot(ax=axs[1], label=f'{name} H1')
    #axs[1].set_label(f'Urban {name} H1')

    #bdat["H1"].plot(ax=figH1)
    #figH1.set_label(f'Urban {name} H1')

    ## Homo 2
    #bdat["H2"].plot(ax=axs[2])
    #fdat = sr.bc_to_sr(bdat, degree="H2")

    #fdat.plot(ax=axs[2], label=f'{name} H2')
    #axs[2].set_label()

def visHN(data: ndarray, name: Union[str, float, int], axs: ndarray=None):

    datadist = sr.Distance(squareform(pdist(data, metric="jaccard")))
    #plot_diagrams(rp(datadist, maxdim=2, distance_matrix=True)['dgms'], show=True)

    if axs is None:
        fig, axs = plt.subplots(3, 1)

    ## Homo 0
    maxdim = 2
    bdat = datadist.get_bc(maxdim=maxdim)

    fdat = sr.bc_to_sr(bdat, degree="H0")

    fdat.plot(ax=axs[0], label=f'{name} H0')
    #axs[0].set_label(f'Urban {name} H0')


    ## Homo 1
    fdat = sr.bc_to_sr(bdat, degree="H1")

    fdat.plot(ax=axs[1], label=f'{name} H1')
    #axs[1].set_label(f'Urban {name} H1')

    #bdat["H1"].plot(ax=figH1)
    #figH1.set_label(f'Urban {name} H1')

    ## Homo 2
    fdat = sr.bc_to_sr(bdat, degree="H2")

    fdat.plot(ax=axs[2], label=f'{name} H2')
    #axs[2].set_label()


## helper func
def __RandNPointsHomoLoop(data: Union[ndarray, list]):

    #diagrams = rp(curbdist[0], maxdim=2, thresh= 4, distance_matrix=True)['dgms']
    stuff = {"H0": empty((3,0)), "H1": empty((3,0)), "H2": empty((3,0))}
    summize = 0
    for idx, cp in enumerate(data):
        for idx2, H_level in enumerate(_data2sr(cp)):
            stuff[f"H{idx2}"] = concatenate([stuff[f"H{idx2}"], concatenate([array([idx+1] * H_level.shape[1])[None, :], H_level], axis=0)], axis=1)

        summize += cp.shape[0]


    diff = summize-stuff["H0"].shape[1]
    if diff > 0:
        stuff["H0"] = concatenate([zeros((3, diff)), stuff["H0"]], axis=1)
    elif diff < 0:
        stuff["H0"] = stuff["H0"][:,:summize]

    return stuff


## Create homology from x sampled closest points
def RandNPointsHomo(data: Union[ndarray, list]):
    ## choose random point, sort by the closest points and choose the 100 closest points.
    #fig, axs = plt.subplots(3, 1)

    return __RandNPointsHomoLoop(data)


def visH0sr(data: ndarray):
    ## Create matrix distance
    DDAT = sr.Distance(squareform(pdist(data, metric="euclidean")))

    ## anoter stablerank method for H0 homo
    clustering_method = "complete"
    fnat = DDAT.get_h0sr(clustering_method=clustering_method)

    fnat.plot()