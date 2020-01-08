from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
import time
import math

from occupancyGrid import OccupancyGrid

def get2DPose(shape):
  return (shape.get_position()[:2] + shape.get_orientation()[-1:])

def set2DPose(shape, pose):
      """Sets the 2D (top-down) pose of the robot [x, y, yaw]
      :param pose: A List containing the x, y, yaw (in radians).
      """
      x, y, yaw = pose
      shape.set_position([x, y, shape.get_position()[2]])
      shape.set_orientation([0, 0, yaw])

def setupOccGrid(occ_grid, vision_sensor):
  # Capture vision depth and create occupancy grid
  pr.step()
  depth = vision_sensor.capture_depth()
  occ_grid.fromDepth(depth)

def update_coverage(occ_grid, vision_sensor):
  depth = vision_sensor.capture_depth()
  new_grid = OccupancyGrid()
  new_grid.fromDepth(depth)

  occ_grid.grid = np.maximum(occ_grid.grid, new_grid.grid)

# Get Scene
SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_cpp.ttt')

# Set numpy printing options
np.set_printoptions(threshold=(100*100), formatter={'all':lambda x: str(x) + ','})

# Setup occ_grid
occ_grid = OccupancyGrid()

# Start Simulation
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
robot = Shape('robot')
vision_sensor = VisionSensor('vision_sensor')

bounding_box = robot.get_bounding_box()
bounding_box_in_cells = []

for i in range(2):
  min_val = bounding_box[2*i]
  max_val = bounding_box[2*i+1]
  bounding_box_in_cells.append(occ_grid.worldToMap(min_val,max_val))

size_x = abs(bounding_box_in_cells[0][0] - 50) + abs(bounding_box_in_cells[0][1] - 50)
size_y = abs(bounding_box_in_cells[1][0] - 50) + abs(bounding_box_in_cells[1][1] - 50)

# Place at top left edge
start_mx = size_x/2 - 0.5
start_my = size_y/2 - 0.5

start_pose = occ_grid.mapToWorld(start_mx, start_my) + (0,)
set2DPose(robot, start_pose)

setupOccGrid(occ_grid, vision_sensor)

current_pose = list(start_pose)

start_time = time.time()

'''
# Boustrophedon method
# Time = 15.6, Coverage = 100%

for i in range(int(100 / size_x)):
  for j in range(int(99 - size_y/2)):
    multiplier = 1 if i % 2 == 0 else -1

    current_pose[1] += multiplier * 0.01
    set2DPose(robot, current_pose)
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)

  for k in range(size_x):
    current_pose[0] += 0.01
    set2DPose(robot, current_pose)
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)
'''

'''
# Boustrophedon Method w/ Sweep
# Time = 17.7, Coverage = 92%

r = math.sqrt((size_x/2)**2 + (size_y/2)**2) * 0.01
theta = math.radians(90)
step_size = 2

# Initial movement
for j in range(size_x):
    current_pose[1] += 0.01
    set2DPose(robot, current_pose)
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)

for i in range(int(100 / size_x) - 1):
  for j in range(int(100 - 2*size_x - size_y/2)):
    multiplier = 1 if i % 2 == 0 else -1

    current_pose[1] += multiplier * 0.01
    set2DPose(robot, current_pose)
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)

  # Sweep the block
  for k in range(0,180,step_size):
    rad = math.radians(step_size)
    current_pose[0] += r * (math.sin(rad + theta) - math.sin(theta))
    current_pose[1] -= r * (math.cos(rad + theta) - math.cos(theta))
    theta += -multiplier * rad
    current_pose[2] += -multiplier * rad

    set2DPose(robot, current_pose)
    
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)

for j in range(int(99 - size_x - size_y/2)):
    current_pose[1] += 0.01
    set2DPose(robot, current_pose)
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)
'''

# Spiral Method

for i in range(int(100 / size_x)):

  # Horizontal movement
  for j in range(int(99 - size_y/2 - (size_y/2)*i)):

    current_pose[1] += 0.01
    set2DPose(robot, current_pose)
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)

  # Vertical movement
  for k in range(int(100 - size_x - size_x*i)):
    current_pose[0] += 0.01
    set2DPose(robot, current_pose)
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)

  # Horizontal movement
  for j in range(int(99 - size_y/2 - (size_y/2)*i)):

    current_pose[1] -= 0.01
    set2DPose(robot, current_pose)
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)

  # Vertical movement
  for k in range(int(100 - size_x - size_x*(i+1))):
    current_pose[0] -= 0.01
    set2DPose(robot, current_pose)
    pr.step()
    update_coverage(occ_grid, vision_sensor)
    time.sleep(0.01)

end_time = time.time()
coverage_percent = np.count_nonzero(occ_grid.grid) / 100
print("Time elapsed: " + str(end_time - start_time))
print("Coverage = " + str(coverage_percent))




# print(occ_grid)

# Get start state
# Get goal state

# End Simulation
pr.stop()
pr.shutdown()