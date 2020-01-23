from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
import time
import math

from occupancyGrid import OccupancyGrid

class Pose:
  def __init__(self, x, y, theta):
    self.x = x
    self.y = y
    self.theta = theta

  def __str__(self):
    return '{}, {}, {}'.format(self.x, self.y, self.theta)

  def __add__(self, rhs):
    return Pose(self.x + rhs.x, self.y + rhs.y, self.theta + rhs.theta)

def get2DPose(shape):
  return (shape.get_position()[:2] + shape.get_orientation()[-1:])

def set2DPose(shape, pose):
      '''
      Sets the 2D (top-down) pose of the robot [x, y, yaw]
      '''

      x = pose.x
      y = pose.y
      yaw = pose.theta
      shape.set_position([x, y, shape.get_position()[2]])
      shape.set_orientation([0, 0, yaw])

def setupOccGrid(occ_grid, vision_sensor):
  # Capture vision depth and create occupancy grid
  pr.step()
  depth = vision_sensor.capture_depth()
  occ_grid.fromDepth(depth)

def boustrophedon(block_size_x, block_size_y, area_size_x, area_size_y, occ_grid):

  max_block_dim = max(block_size_x, block_size_y)
  min_block_dim = min(block_size_x, block_size_y)

  max_area_dim = max(area_size_x, area_size_y)
  min_area_dim = min(area_size_x, area_size_y)

  is_block_max_x = False
  is_area_max_x = False
  move_in_x_dir = False

  if max_block_dim == block_size_x:
    is_block_max_x = True
  
  if max_area_dim == area_size_x:
    is_area_max_x = True
    move_in_x_dir = True

  path = []

  # TODO: Need to offset from centre
  start_position = occ_grid.mapToWorld(max_block_dim/2 - 0.5, min_block_dim/2 - 0.5)
  current_pose = Pose(start_position[0], start_position[1], 0)
  
  # If the max of block and area are in the same direction then we need to rotate by 90 deg
  if (is_block_max_x and is_area_max_x) or ((not is_block_max_x and not is_area_max_x)):
    current_pose = Pose(start_position[1], start_position[0], math.radians(90))

  # TODO:
  # Check if block can fit in area -> block needs to be spun
  # Check if block can perfectly sweep area -> else block needs to be spun

  sweeps = min_area_dim / max_block_dim
  moves_per_sweep = max_area_dim - min_block_dim

  path.append(current_pose)
  multiplier = 1

  # Boustrophedon method
  for i in range(int(sweeps)):
    multiplier = 1 if i % 2 == 0 else -1

    for j in range(int(moves_per_sweep)):

      if move_in_x_dir:
        current_pose += Pose(multiplier * 0.01, 0, 0)
      else:
        current_pose += Pose(0, multiplier * 0.01, 0)
        
      path.append(current_pose)

    # Break the loop so we dont perform the shift at the end
    if i == int(sweeps) - 1:
      continue

    for k in range(max_block_dim):
      if move_in_x_dir:
        current_pose += Pose(0, 0.01, 0)
      else:
        current_pose += Pose(0.01, 0, 0)

      path.append(current_pose)

  if int(sweeps) != sweeps:
    remaining = min_area_dim - (int(sweeps) * max_block_dim)
    print(sweeps)
    print(max_block_dim)
    print(min_area_dim)
    print(remaining)

    for i in range(remaining):
      if move_in_x_dir:
        current_pose += Pose(0, 0.01, 0)
      else:
        current_pose += Pose(0.01, 0, 0)

      path.append(current_pose)

    for j in range(int(moves_per_sweep)):

      if move_in_x_dir:
        current_pose += Pose(-multiplier * 0.01, 0, 0)
      else:
        current_pose += Pose(0, -multiplier * 0.01, 0)
        
      path.append(current_pose)

  return path

# Get Scene
SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_cpp.ttt')

# Start Simulation
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
robot = Shape('robot')
vision_sensor = VisionSensor('vision_sensor')

# Setup occ_grid
occ_grid = OccupancyGrid()
setupOccGrid(occ_grid, vision_sensor)

path = boustrophedon(20, 2, 100,100, occ_grid)

for p in path:
  set2DPose(robot, p)
  pr.step()
  time.sleep(0.01)

# End Simulation
pr.stop()
pr.shutdown()