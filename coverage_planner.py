from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.octree import Octree
from pyrep.const import PrimitiveShape
import numpy as np
import time
import math

from occupancyGrid import OccupancyGrid
from gridSubmapper import GridSubmapper
from submapPlanner import SubmapPlanner

from utils import Pose
from testing_tools import random_map_generator

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



# Set numpy printing options
np.set_printoptions(threshold=100*100, formatter={'all':lambda x: str(x) + ','})

# Get Scene
SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_cpp.ttt')

# Start Simulation
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
robot = Shape('start_pose')
octree = Octree('Octree')
vision_sensor = VisionSensor('vision_sensor')

random_map_generator.generate_random_map(3, rng_seed=4143)

# Setup occ_grid
occ_grid = OccupancyGrid()
setupOccGrid(occ_grid, vision_sensor)

pr.step()

submapper = GridSubmapper(occ_grid)
submapper.process()

bounding_box = robot.get_bounding_box()

block_size_x = int(round(bounding_box[1] - bounding_box[0], 3)/occ_grid.resolution)
block_size_y = int(round(bounding_box[3] - bounding_box[2], 3)/occ_grid.resolution)

planner = SubmapPlanner(occ_grid, block_size_x, block_size_y)
path = planner.process(submapper.submaps)

# print(path)
prev_rot = 0

for p in path:
  wx, wy = occ_grid.mapToWorld(p[0], p[1])
  pose = Pose(wx, wy, math.radians(p[2] * 90))
  # print(wx, wy)
  set2DPose(robot, pose)


  # Visualize the coverage
  if p[2] != prev_rot:
    prev_rot = p[2]
    block_size_x, block_size_y = block_size_y, block_size_x

  for i in range(-block_size_x//2 + 1, block_size_x//2):
    for j in range(-block_size_y//2 + 1, block_size_y//2):
      px = p[0] + i
      py = p[1] + j

      if not octree.check_point_occupancy([px, py, 0]):
        nx, ny = occ_grid.mapToWorld(px, py)
        octree.insert_voxels([nx, ny, 0], [255,0,0])

  pr.step()
  time.sleep(0.01)

time.sleep(1)


# visualization_grid = submapper.visualization()
# print(visualization_grid)
# print()
# print(planner.inflated_occ_grid[0])
# print()
# print(planner.inflated_occ_grid[1])

# End Simulation
pr.stop()
pr.shutdown()
