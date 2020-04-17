import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
#from sklearn.preprocessing import normalize
import matplotlib.cm as cm

# Procedural algorithm for the generation of two-dimensional Poission-disc
# sampled ("blue") noise. For mathematical details, please see the blog
# article at https://scipython.com/blog/poisson-disc-sampling-in-python/
# Christian Hill, March 2017.

# Choose up to k points around each reference point as candidates for a new
# sample point
k = 30

# Minimum distance between samples
r = 1.7 #1.7 #0.5 #1.0 #1.7

width, height = 60, 45

# Cell side length
a = r/np.sqrt(2)
# Number of cells in the x- and y-directions of the grid
nx, ny = int(width / a) + 1, int(height / a) + 1

# A list of coordinates in the grid of cells
coords_list = [(ix, iy) for ix in range(nx) for iy in range(ny)]
# Initilalize the dictionary of cells: each key is a cell's coordinates, the
# corresponding value is the index of that cell's point's coordinates in the
# samples list (or None if the cell is empty).
cells = {coords: None for coords in coords_list}

def get_cell_coords(pt):
    """Get the coordinates of the cell that pt = (x,y) falls in."""

    return int(pt[0] // a), int(pt[1] // a)

def get_neighbours(coords):
    """Return the indexes of points in cells neighbouring cell at coords.

    For the cell at coords = (x,y), return the indexes of points in the cells
    with neighbouring coordinates illustrated below: ie those cells that could 
    contain points closer than r.

                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo

    """
    
    dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
            (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
            (-1,2),(0,2),(1,2),(0,0)]
    neighbours = []
    for dx, dy in dxdy:
        neighbour_coords = coords[0] + dx, coords[1] + dy
        if not (0 <= neighbour_coords[0] < nx and
                0 <= neighbour_coords[1] < ny):
            # We're off the grid: no neighbours here.
            continue
        neighbour_cell = cells[neighbour_coords]
        if neighbour_cell is not None:
            # This cell is occupied: store this index of the contained point.
            neighbours.append(neighbour_cell)
    return neighbours

def point_valid(pt):
    """Is pt a valid point to emit as a sample?

    It must be no closer than r from any other point: check the cells in its
    immediate neighbourhood.

    """

    cell_coords = get_cell_coords(pt)
    for idx in get_neighbours(cell_coords):
        nearby_pt = samples[idx]
        # Squared distance between or candidate point, pt, and this nearby_pt.
        distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
        if distance2 < r**2:
            # The points are too close, so pt is not a candidate.
            return False
    # All points tested: if we're here, pt is valid
    return True
    
def get_point(k, refpt):
    """Try to find a candidate point relative to refpt to emit in the sample.

    We draw up to k points from the annulus of inner radius r, outer radius 2r
    around the reference point, refpt. If none of them are suitable (because
    they're too close to existing points in the sample), return False.
    Otherwise, return the pt.

    """
    i = 0
    while i < k:
        rho, theta = np.random.uniform(r, 2*r), np.random.uniform(0, 2*np.pi)
        pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
        if not (0 < pt[0] < width and 0 < pt[1] < height):
            # This point falls outside the domain, so try again.
            continue
        if point_valid(pt):
            return pt
        i += 1
    # We failed to find a suitable point in the vicinity of refpt.
    return False

class UniformNoise():
    """A class for generating uniformly distributed, 2D noise."""

    def __init__(self, width=50, height=50, n=None):
        """Initialise the size of the domain and number of points to sample."""

        self.width, self.height = width, height
        if n is None:
            n = int(width * height)
        self.n = n

    def reset(self):
        pass

    def sample(self):
        return np.array([np.random.uniform(0, width, size=self.n),
                         np.random.uniform(0, height, size=self.n)]).T

n = int(width * height / np.pi / a**2)
uniform_noise = UniformNoise(width, height, n) 
print(uniform_noise.sample().shape)
uni_x = uniform_noise.sample()[:,0]
uni_y = uniform_noise.sample()[:,1]       

# Pick a random point to start with.
pt = (np.random.uniform(0, width), np.random.uniform(0, height))
samples = [pt]
# Our first sample is indexed at 0 in the samples list...
cells[get_cell_coords(pt)] = 0
# ... and it is active, in the sense that we're going to look for more points
# in its neighbourhood.
active = [0]

nsamples = 1
# As long as there are points in the active list, keep trying to find samples.
while active:
    # choose a random "reference" point from the active list.
    idx = np.random.choice(active)
    refpt = samples[idx]
    # Try to pick a new point relative to the reference point.
    pt = get_point(k, refpt)
    if pt:
        # Point pt is valid: add it to the samples list and mark it as active
        samples.append(pt)
        nsamples += 1
        active.append(len(samples)-1)
        cells[get_cell_coords(pt)] = len(samples) - 1
    else:
        # We had to give up looking for valid points near refpt, so remove it
        # from the list of "active" points.
        active.remove(idx)
#print(samples)
def column(matrix, i):
    return [row[i] for row in matrix]

datax = column(samples,0)
datay = column(samples,1)
data = np.array(samples)

kde_uni = stats.gaussian_kde(uniform_noise.sample().T)
density_uni = kde_uni(uniform_noise.sample().T)
#normalize_density = density/max(density)

kde = stats.gaussian_kde(data.T)
density = kde(data.T)
normalize_density = density/max(density)



cmap =  cm.jet #cm.hot #'Blues'


counts, xedges, yedges = np.histogram2d(datax, datay, bins=(60, 45))
#print(counts.shape)
#print(np.amax(counts))
#print(counts)
xidx = np.clip(np.digitize(datax, xedges), 0, counts.shape[0]-1)
yidx = np.clip(np.digitize(datay, yedges), 0, counts.shape[1]-1)
CC = counts[xidx, yidx]
#CC = counts

#print(data)
plt.figure(1)
plt.scatter(data[:,0],data[:,1], c=normalize_density,  s=1, cmap=cmap)
plt.colorbar()
plt.xlim(0, width)
plt.ylim(0, height)
plt.axis('off')
plt.interactive(True)
plt.show()


plt.figure(2)
plt.scatter(data[:,0],data[:,1], c=density,  s=1, cmap=cmap)
plt.colorbar()
plt.xlim(0, width)
plt.ylim(0, height)
plt.axis('off')
plt.show()

plt.figure(3)
plt.hist2d(data[:,0],data[:,1], density = True, bins=(60, 45),cmap=cmap) #density = True??
plt.colorbar()
plt.show()


plt.figure(4)
plt.scatter(datax,datay, c=CC,  s=1, cmap=cmap)
plt.colorbar()
plt.xlim(0, width)
plt.ylim(0, height)
plt.axis('off')
plt.show()

plt.figure(5)
plt.hexbin(datax, datay, gridsize=(60, 45), cmap=cmap)
plt.colorbar()
plt.show()

plt.figure(6)
plt.hist2d(uni_x,uni_y, density = True, bins=(60, 45),cmap=cmap) #density = True??
plt.colorbar()
plt.show()

plt.figure(7)
plt.scatter(uni_x,uni_y, color='r', s=5, alpha=0.6, lw=0)
#plt.scatter(*zip(*samples), color='r', alpha=0.6, lw=0)
plt.xlim(0, width)
plt.ylim(0, height)
plt.axis('off')
plt.show()

plt.figure(8)
plt.scatter(uni_x,uni_y, c=density_uni,  s=1, cmap=cmap)
#plt.scatter(datax[:min(len(uni_x),len(datax))],datay[:min(len(uni_y),len(datay))], c=density_uni,  s=1, cmap=cmap)
plt.colorbar()
plt.xlim(0, width)
plt.ylim(0, height)
plt.axis('off')
plt.show()



plt.figure(9)
plt.scatter(datax,datay, color='r', s=5, alpha=0.6, lw=0)
#plt.scatter(*zip(*samples), color='r', alpha=0.6, lw=0)
plt.xlim(0, width)
plt.ylim(0, height)
plt.axis('off')
plt.savefig('poisson.png')
plt.interactive(False)
plt.show()
