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

# For visualization
coverage_grid = OccupancyGrid()
setupOccGrid(coverage_grid, vision_sensor)

obstacle_mask = coverage_grid.grid == 0
mask = obstacle_mask

dc_account_for = set()

for p in path:

  # Move the robot along the path
  wx, wy = occ_grid.mapToWorld(p[0], p[1])
  pose = Pose(wx, wy, math.radians(p[2] * 90))
  # print(wx, wy)
  set2DPose(robot, pose)

  # Visualization
  new_coverage = OccupancyGrid()
  setupOccGrid(new_coverage, vision_sensor)
  prev_grid = np.array(coverage_grid.grid, copy=True)
  coverage_grid.grid[mask] += new_coverage.grid[mask]
  
  pos = np.argwhere((coverage_grid.grid - prev_grid) == 1)

  new_mask = new_coverage.grid == 0
  mask = np.logical_and(obstacle_mask, new_mask)

  if len(pos) == 0:
    continue
  
  min_x = np.min(pos[:,0])
  min_y = np.min(pos[:,1])
  max_x = np.max(pos[:,0])
  max_y = np.max(pos[:,1])

  centre_x = (max_x + min_x)/2
  centre_y = (max_y + min_y)/2

  wx, wy = coverage_grid.mapToWorld(centre_y, centre_x)

  grid_length = max_x - min_x
  grid_width =  max_y - min_y

  # Pad length and width with resolution to prevent issues when one is zero and to ensure no gaps between shapes
  length = coverage_grid.resolution * grid_length + coverage_grid.resolution
  width = coverage_grid.resolution * grid_width + coverage_grid.resolution

  Shape.create(type=PrimitiveShape.CUBOID, size=[width, length,.05], position=[wx,wy,0.0], color=[255,0,0], static=True, renderable=False)

  pr.step()
  time.sleep(0.01)

# print(coverage_grid)

# visualization_grid = submapper.visualization()
# print(visualization_grid)
# print()
# print(planner.inflated_occ_grid[0])
# print()
# print(planner.inflated_occ_grid[1])

# End Simulation
pr.stop()
pr.shutdown()
