from loadData import loadNatural, loadUrban
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.spatial.distance import cdist, pdist, squareform
import stablerank.srank as sr
from numpy import shape, min, max, mean, where, argsort, zeros
from numpy.random import choice

from ripser import ripser as rp
from persim import plot_diagrams

sDir = Path(__file__).parent

urbPath = sDir / Path('./data/urban2012-01-08/velodyne_sync/1326036605234924.bin')

urb = loadUrban(urbPath)
#urb = urb[:20000, :]
# Center the points w.r.t. average
#urb = urb - urb.mean(axis=0)

# Radii of 4 meter from center
#surb = urb[:,0]**2 * urb[:,1]**2 < 4
#urb = urb[surb]

#print("urban", shape(urb))
#print(urb.max(axis=0))
#print(urb.min(axis=0))

# choose random point, sort by the closest points and choose the 100 closest points.
nump = 3
npoint = 1000

rurb = choice(shape(urb)[0], nump)

curb = zeros((nump, npoint, 3))

for idx, sample in enumerate(rurb):
    tmp = urb[argsort(cdist(urb, urb[None, sample]).squeeze())]
    curb[idx] = tmp[:npoint]



natPath = sDir / Path('./data/outdoorBox_met/box_met_000/box_met_000.xyz')

nat = loadNatural(natPath)
#nat = nat[:20000, :]
# Center the points w.r.t. average
#nat = nat - nat.mean(axis=0)

# Radii of 4 meter from center
#snat = nat[:,0]**2 * nat[:,1]**2 < 4
#nat = nat[snat]

#print("natural", shape(nat))
#print(nat.max(axis=0))
#print(nat.min(axis=0))

# choose random point, sort by the closest points and choose the 100 closest points.

rnat = choice(shape(nat)[0], nump)

cnat = zeros((nump, npoint, 3))

for idx, sample in enumerate(rnat):
    tmp = nat[argsort(cdist(nat, nat[None, sample]).squeeze())]
    cnat[idx] = tmp[:npoint]


fig = plt.figure()
ax = fig.add_subplot(111)
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
curbdist = []
for idx, cp in enumerate(curb):
    curbdist.append(sr.Distance(squareform(pdist(cp, metric="euclidean"))))

cnatdist = []
for idx, cp in enumerate(cnat):
    cnatdist.append(sr.Distance(squareform(pdist(cp, metric="euclidean"))))


#diagrams = rp(curbdist[0], maxdim=2, thresh= 4, distance_matrix=True)['dgms']
for idx, cp in enumerate(curbdist):
    #plot_diagrams(rp(cp, maxdim=2, distance_matrix=True)['dgms'], show=True)

    maxdim = 2
    burb = cp.get_bc(maxdim=maxdim)

    furb = sr.bc_to_sr(burb, degree="H0")

    purb = furb.plot()
    purb[0].set_label(f'Urban {idx} H0')

    #furb = sr.bc_to_sr(burb, degree="H1")

    #purb = furb.plot(label=f'Urban {idx} H1')
    #purb[0].set_label(f'Urban {idx} H1')


    #furb = sr.bc_to_sr(burb, degree="H2")

    #purb = furb.plot(label=f'Urban {idx} H2')
    #purb[0].set_label(f'Urban {idx} H2')


for idx, cp in enumerate(cnatdist):
    #plot_diagrams(rp(cp, maxdim=2, distance_matrix=True)['dgms'], show=True)

    maxdim = 2
    bnat = cp.get_bc(maxdim=maxdim)

    fnat = sr.bc_to_sr(bnat, degree="H0")

    pnat = fnat.plot()
    pnat[0].set_label(f'Natural {idx} H0')


    #fnat = sr.bc_to_sr(bnat, degree="H1")

    #pnat = fnat.plot(label=f'Natural {idx} H1')
    #pnat[0].set_label(f'Natural {idx} H1')


    #fnat = sr.bc_to_sr(bnat, degree="H2")

    #pnat = fnat.plot(label=f'Natural {idx} H2')
    #pnat[0].set_label(f'Natural {idx} H2')


fig.legend()

plt.show()


#clustering_method = "complete"
#fnat = DNAT.get_h0sr(clustering_method=clustering_method)

#fnat.plot()


#ax.scatter(urb[:,0], urb[:,1], urb[:,2], marker='o')
#ax.scatter(nat[:,0], nat[:,1], nat[:,2], marker='^')

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

#plt.show()