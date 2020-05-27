import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt


class OccupancyGrid:
  occ_grid_converter = np.vectorize(lambda x : 0.0 if x >= 0.98 else 1.0)

  def __init__(self, origin_x = -0.5, origin_y = -0.5):
    self.grid = np.zeros(1)
    self.origin_x = origin_x
    self.origin_y = origin_y
    self.size_x = 100
    self.size_y = 100
    self.resolution = 0.01 # m per pixel

  def clone(self):
    result = OccupancyGrid()
    result.__dict__ = self.__dict__.copy()
    result.grid = np.copy(self.grid)

    return result
    
  def fromDepth(self, depth_data):
    self.grid = self.occ_grid_converter(depth_data)
    self.size_y, self.size_x = self.grid.shape

  def generateRectilinearOcc(self):
    # Convert the occgrid into a uint8 grid
    img_grey = np.uint8(self.grid * 255)

    # Find the contours of the grid
    contours, hierarchy = cv2.findContours(img_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the min and max of x/y coordinates and create a rectangle from min x/y to max x/y
    for shape in contours:
      points = shape.reshape(len(shape), 2)
      max_x, max_y = np.amax(points, axis=0)
      min_x, min_y = np.amin(points, axis=0)

      corners = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
      cv2.rectangle(img_grey, (min_x, min_y), (max_x, max_y), 255, -1)

    ret, rectilinear_grid = cv2.threshold(img_grey, 1, 1, cv2.THRESH_BINARY)

    result = self.clone()
    result.grid = np.float64(rectilinear_grid)

    return result

  def __getitem__(self, key):
    return self.grid[key[1], key[0]]

  def __setitem__(self, key, value):
    self.grid[key[1], key[0]] = value

  def worldToMap(self, wx, wy):
    # return (int((round(wx+0.00001,4) - self.origin_x)/self.resolution), self.size_y - int((round(wy+0.00001,4) - self.origin_y)/self.resolution))
    return (((wx+0.00001) - self.origin_x)//self.resolution, self.size_y - ((wy+0.00001) - self.origin_y)//self.resolution)

  # TODO: This calculation is wrong -> increasing resolution gives incorrect position
  def mapToWorld(self, mx, my):
    return ((mx+1) * self.resolution + self.origin_x, (self.size_y - my) * self.resolution + self.origin_y)

  def inBounds(self, x, y):
    if x >= 0 and x < self.size_x and y >= 0 and y < self.size_y:
      return True
    else:
      return False

  def __str__(self):
    result = ""
    for j in range(self.size_y):
      for i in range(self.size_x):
        result += str(self.grid[j, i])
        result += ', '
      result += '\n'

    return result

  def setup_drawing(self):
    # Set the colormap for drawing
    base_cmap = plt.cm.Reds

    cmaplist = ['k', 'w'] + [base_cmap(i) for i in range(20,255,15)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(cmaplist))

    # define the bins and normalize
    bounds = np.linspace(-1, len(cmaplist) - 2, len(cmaplist))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    self.fig = plt.figure(figsize=(6,12))
    spec = mpl.gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2,1])
    self.ax0 = self.fig.add_subplot(spec[0])
    self.ax1 = self.fig.add_subplot(spec[1])

    self.im = self.ax0.imshow(self.grid, cmap=cmap, norm=norm)
    self.line, = self.ax1.plot(0, 0)

    self.ax0.set_xticklabels([])
    self.ax0.set_yticklabels([])
    self.ax0.title.set_text('Coverage Visualization')

    self.ax1.set_ylim([0, 100])
    self.ax1.set_xticklabels([])
    self.ax1.title.set_text('Coverage v.s. Time')
    self.ax1.set_ylabel('Coverage %')

    self.t = 0

    # self.fig.colorbar(self.im, ax=self.ax0)
    plt.show(block=False)

  def draw(self):
    self.im.set_data(self.grid)

    self.t += 1
    covered = np.count_nonzero(self.grid > 0)
    
    # Dont create a line for every timestep -> will become very slow
    if self.t % 8 == 0:
      self.line, = self.ax1.plot(self.t,covered/(self.size_x * self.size_y) * 100, 'bo')

    self.fig.canvas.draw()
    self.ax0.draw_artist(self.ax0.patch)
    self.ax1.draw_artist(self.line)
