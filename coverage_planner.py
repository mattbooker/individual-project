import math
from os.path import abspath, dirname, join

import numpy as np
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from gridSubmapper import GridSubmapper
from occupancyGrid import OccupancyGrid
from submapPlanner import SubmapPlanner
from testing_tools import random_map_generator
from utils import Pose

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
SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_cpp.ttt')

# Start Simulation
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
robot = Shape('start_pose')
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

# Set obstacles to -1 for coverage calculations
coverage_grid.grid *= -1

coverage_grid.setup_drawing()

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
  
  new_mask = new_coverage.grid == 0
  mask = np.logical_and(obstacle_mask, new_mask)

  coverage_grid.draw()
  pr.step()

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
