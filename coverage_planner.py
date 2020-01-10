from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
import time
import math

from occupancyGrid import OccupancyGrid
from gridSubmapper import GridSubmapper


def setupOccGrid(occ_grid, vision_sensor):
  # Capture vision depth and create occupancy grid
  pr.step()
  depth = vision_sensor.capture_depth()
  occ_grid.fromDepth(depth)

# Get Scene
SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_maze.ttt')

# Set numpy printing options
np.set_printoptions(threshold=(100*100), formatter={'all':lambda x: str(x) + ','})

# Setup occ_grid
occ_grid = OccupancyGrid()

# Start Simulation
pr = PyRep()
pr.launch(SCENE_FILE, headless=True)
pr.start()
# robot = Shape('robot')
vision_sensor = VisionSensor('vision_sensor')

setupOccGrid(occ_grid, vision_sensor)

occ_grid.grid = np.zeros((4000, 4000))
occ_grid.size_x = occ_grid.size_y = 4000

pr.step()

submapper = GridSubmapper(occ_grid)
submapper.process()

# visualization_grid = submapper.visualization()
# print(visualization_grid)

# End Simulation
pr.stop()
pr.shutdown()
