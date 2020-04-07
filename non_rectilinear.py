from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import PrimitiveShape
import numpy as np
import time
import math

from occupancyGrid import OccupancyGrid

import cv2


def setupOccGrid(occ_grid, vision_sensor):
  # Capture vision depth and create occupancy grid
  pr.step()
  depth = vision_sensor.capture_depth()
  occ_grid.fromDepth(depth)

def getCorners(img):
  corners = []
  size_x, size_y = img.shape

  # TODO: test edges of img

  # TODO: Doesnt work for non-rectilinear objects

  for y in range(0, size_y - 1):
    for x in range(0, size_x - 1):
      if img[y, x] == 0:
        continue

      occupied = [bool(img[y, x-1]), bool(img[y-1, x]), bool(img[y, x+1]), bool(img[y+1, x])]

      if sum(occupied) == 2:
        if (occupied[0] and occupied[2]) or (occupied[1] and occupied[3]):
          continue
        else:
          corners.append((x, y))
  
  return corners

# Set numpy printing options
np.set_printoptions(threshold=100*100, formatter={'all':lambda x: str(x) + ','})

# Get Scene
SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_nonrectilinear.ttt')

# Start Simulation
pr = PyRep()
pr.launch(SCENE_FILE, headless=True)
pr.start()

# Setup occ_grid
vision_sensor = VisionSensor('vision_sensor')
occ_grid = OccupancyGrid()
setupOccGrid(occ_grid, vision_sensor)
pr.step()

# End Simulation
pr.stop()
pr.shutdown()

# Process
img_grey = np.uint8(occ_grid.grid * 255)

contours, hierarchy = cv2.findContours(img_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_color = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)
intermediate_step = img_color.copy()

blank = np.zeros((len(img_grey), len(img_grey)), dtype=np.uint8)
cv2.drawContours(blank, contours, -1, (255,255,255), 1, cv2.LINE_AA)

for shape in contours:
  points = shape.reshape(len(shape), 2)
  max_x, max_y = np.amax(points, axis=0)
  min_x, min_y = np.amin(points, axis=0)

  corners = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
  cv2.rectangle(intermediate_step, (min_x, min_y), (max_x, max_y), (0,0,255), 1)
  cv2.rectangle(img_color, (min_x, min_y), (max_x, max_y), (0,0,255), -1)

cv2.imshow("Original", img_grey)
cv2.imshow("Intermediate Step", intermediate_step)
cv2.imshow("Rectilinear", img_color)

cv2.waitKey(0)
cv2.destroyAllWindows()

