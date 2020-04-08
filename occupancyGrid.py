import numpy as np
import cv2

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
    return (mx * self.resolution + self.origin_x, (self.size_y - my) * self.resolution + self.origin_y)

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