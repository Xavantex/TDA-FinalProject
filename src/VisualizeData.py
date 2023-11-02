from loadData import loadNatural, loadUrban
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.spatial.distance import cdist, pdist, squareform
import stablerank.srank as sr
from numpy import shape, min, max, mean, where

from ripser import ripser as rp

sDir = Path(__file__).parent

urbPath = sDir / Path('./data/urban2012-01-08/velodyne_sync/1326036605234924.bin')

urb = loadUrban(urbPath)
#urb = urb[:20000, :]
# Center the points w.r.t. average
urb = urb - urb.mean(axis=0)
# Radii of 4 meter from center
surb = urb[:,0]**2 * urb[:,1]**2 > 16
urb = urb[surb]
#print(urb.max(axis=0))
#print(urb.min(axis=0))


natPath = sDir / Path('./data/outdoorBox_met/box_met_000/box_met_000.xyz')

nat = loadNatural(natPath)
#nat = nat[:20000, :]
# Center the points w.r.t. average
nat = nat - nat.mean(axis=0)
# Radii of 4 meter from center
snat = nat[:,0]**2 * nat[:,1]**2 > 16
nat = nat[snat]
#print(nat.max(axis=0))
#print(nat.min(axis=0))


#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax = fig.add_subplot(projection='3d')

#ax.scatter(urb[:,0], urb[:,1], urb[:,2], marker='o')
#ax.scatter(nat[:,0], nat[:,1], nat[:,2], marker='^')
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#plt.show()

#DURB = sr.Distance(squareform(pdist(urb, metric="euclidean")))
#DNAT = sr.Distance(squareform(pdist(nat, metric="euclidean")))
DURB = squareform(pdist(urb, metric="euclidean"))
#DNAT = squareform(pdist(nat, metric="euclidean"))


diagrams = rp(DURB, maxdim=2, thresh= 4, distance_matrix=True)['dgms']
print(diagrams)

exit(0)

clustering_method = "complete"

maxdim = 1
bnat = DNAT.get_bc(maxdim=maxdim)
burb = DURB.get_bc(maxdim=maxdim)

#fnat0 = sr.bc_to_sr(bnat, degree="H0")
#furb0 = sr.bc_to_sr(burb, degree="H0")
fnat1 = sr.bc_to_sr(bnat, degree="H1")
furb1 = sr.bc_to_sr(burb, degree="H1")

pnat = fnat1.plot(label='Natural')
purb = furb1.plot(label='Urban')

ax.add_line(pnat[0])
ax.add_line(purb[0])

plt.show()

#fnat = DNAT.get_h0sr(clustering_method=clustering_method)

#fnat.plot()


#ax.scatter(urb[:,0], urb[:,1], urb[:,2], marker='o')
#ax.scatter(nat[:,0], nat[:,1], nat[:,2], marker='^')

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

#plt.show()