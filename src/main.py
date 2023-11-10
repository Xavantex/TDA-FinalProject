from loadData import loadNatural, loadUrban, urbanPath, naturalPath
from processData import center, radiiFilter, sectionData, sampleNclosePoints, xSamples
from VisualizeData import visRAW, visRandNPointsHomo, visH0sr, visHN

################################
## Fetch urban path
urbPath = urbanPath() / '1326036605234924.bin'

## Load urban data
urb = loadUrban(urbPath)

## Center the points w.r.t. average
#urb = center(urb, urb.mean(axis=0))

# filter radii
#urb = radiiFilter(urb, 4)

## choose random point, sort by the closest points and choose the 100 closest points.
nump = 3
cpoint = 500
#curb = sampleNclosePoints(urb, nump, npoint)


natPath = naturalPath() / 'box_met_000/box_met_000.xyz'

nat = loadNatural(natPath)
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
nat=sectionData(nat, sects=4)#, minimum: float=None, maximum: float=None)
urb=sectionData(urb, sects=4)
visRandNPointsHomo(nat, urb)
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