from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import PrimitiveShape
import numpy as np
import time

from occupancyGrid import OccupancyGrid
from gridSubmapper import GridSubmapper
from submapPlanner import SubmapPlanner

def setupOccGrid(occ_grid, vision_sensor):
  # Capture vision depth and create occupancy grid
  pr.step()
  depth = vision_sensor.capture_depth()
  occ_grid.fromDepth(depth)

def set2DPose(shape, pose):
      '''
      Sets the 2D (top-down) pose of the robot [x, y, yaw]
      '''

      x = pose.x
      y = pose.y
      yaw = pose.theta
      shape.set_position([x, y, shape.get_position()[2]])
      shape.set_orientation([0, 0, yaw])


# Get Scene
SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_maze.ttt')

# Set numpy printing options
np.set_printoptions(threshold=100*100, formatter={'all':lambda x: str(x) + ','})

# Start Simulation
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
robot = Shape('start_pose')
vision_sensor = VisionSensor('vision_sensor')

# Setup occ_grid
occ_grid = OccupancyGrid()
setupOccGrid(occ_grid, vision_sensor)

pr.step()

submapper = GridSubmapper(occ_grid)
submapper.process()

planner = SubmapPlanner(occ_grid, 20, 2)
planner.process(submapper.submaps)

for s in submapper.submaps:
  path = planner.generatePathForSubmap(s)

  for p in path:
    set2DPose(robot, p)
    pr.step()
    time.sleep(0.02)

# path = planner.generatePathForSubmap(submapper.submaps[1])

# for p in path:
#   # print(p)
#   set2DPose(robot, p)
#   pr.step()
#   time.sleep(0.05)

# visualization_grid = submapper.visualization()
# print(visualization_grid)

# End Simulation
pr.stop()
pr.shutdown()
