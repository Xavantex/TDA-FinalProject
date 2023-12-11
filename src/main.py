from glob import glob
from numpy import array, empty, concatenate, save as npsave
from pathlib import Path

from loadData import loadNatural, loadUrban, urbanPath, naturalPath
from processData import center, radiiFilter, sectionData, sampleNclosePoints, xSamples
from VisualizeData import visRAW, RandNPointsHomo, visH0sr, visHN

################################
## Fetch urban path
#urbPath = urbanPath() / '1326036605234924.bin'
#natPath = naturalPath() / 'box_met_000/box_met_000.xyz'

numFiles = 112

files = array(glob(str(urbanPath()) + "/*"))
urbFiles = xSamples(files, numFiles)

files = array(glob(str(naturalPath()) + "/*/*.xyz"))
natFiles = xSamples(files, numFiles)

cpoint = 250
tot_data = empty((0, 3, cpoint))
tot_labels = empty((0, 2))
ref_tot_data = empty((0, cpoint, 3))
ref_tot_labels = empty((0, 2))
for urbPath, natPath in zip(urbFiles, natFiles):
    ## Load urban data
    urb = loadUrban(Path(urbPath))

    ## Center the points w.r.t. average
    #urb = center(urb, urb.mean(axis=0))

    # filter radii
    #urb = radiiFilter(urb, 4)

    ## choose random point, sort by the closest points and choose the 100 closest points.
    #nump = 3
    #curb = sampleNclosePoints(urb, nump, npoint)


    nat = loadNatural(Path(natPath))
    ################################


    ################################
    ## Center the points w.r.t. average
    #nat = center(nat, nat.mean(axis=0))
    ################################

    ################################
    ## Radii of 4 meter from center
    #nat = radiiFilter(nat, 4)
    ################################


    ################################
    nat = xSamples(nat, cpoint)
    urb = xSamples(urb, cpoint)

    if not Path("refTotdata.npz").exists():
        ref_tot_data = concatenate([ref_tot_data, nat[None, :, :]], axis=0)
        ref_tot_data = concatenate([ref_tot_data, urb[None, :, :]], axis=0)
        ref_tot_labels = concatenate([ref_tot_labels, array([[1,0], [0,1]])], axis=0)



    if not Path("totdata.npz").exists():
        nat=sectionData(nat, sects=4)#, minimum: float=None, maximum: float=None)
        urb=sectionData(urb, sects=4)

        tot_data = concatenate([tot_data, RandNPointsHomo(nat)["H0"][None, :, :]], axis=0)
        tot_data = concatenate([tot_data, RandNPointsHomo(urb)["H0"][None, :, :]], axis=0)
        tot_labels = concatenate([tot_labels, array([[1,0], [0,1]])], axis=0)

        ################################

        ################################
        #visRAW(nat, urb)
        ################################

        ################################
        #cnat = sampleNclosePoints(nat, nump, cpoint)
        #curb = sampleNclosePoints(urb, nump, cpoint)
        #visRandNPointsHomo(cnat, curb)
        ################################

        ################################
        #visH0sr(nat)

if not Path("totdata.npz").exists():
    with open(Path("labels.npz"), "wb") as file:
        npsave(file, tot_labels, allow_pickle=False)
    with open(Path("totdata.npz"), "wb") as file:
        npsave(file, tot_data, allow_pickle=False)

if not Path("refTotdata.npz").exists():
    with open(Path("refLabels.npz"), "wb") as file:
        npsave(file, ref_tot_labels, allow_pickle=False)
    with open(Path("refTotdata.npz"), "wb") as file:
        npsave(file, ref_tot_data, allow_pickle=False)