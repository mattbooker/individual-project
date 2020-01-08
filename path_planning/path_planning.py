from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
import time
import math

from occupancyGrid import OccupancyGrid
from dijkstra import Dijkstra
from RRT import RRT
from basicCC import basicCC

def get_2d_pose(shape):
  return (shape.get_position()[:2] + shape.get_orientation()[-1:])

def set_2d_pose(shape, pose):
      """Sets the 2D (top-down) pose of the robot [x, y, yaw]
      :param pose: A List containing the x, y, yaw (in radians).
      """
      x, y, yaw = pose
      shape.set_position([x, y, shape.get_position()[2]])
      shape.set_orientation([0, 0, yaw])

# Get Scene
SCENE_FILE = join(dirname(abspath(__file__)), '../scenes/scene_maze.ttt')

# Set numpy printing options
np.set_printoptions(threshold=(100*100), formatter={'all':lambda x: str(x) + ','})

# Setup occ_grid
occ_grid = OccupancyGrid()

# Start Simulation
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
robot = Shape('start_pose')
goal = Shape('goal_pose')
vision_sensor = VisionSensor('vision_sensor')

# Since our "robot" is actually a shape we need to define our own 2d pose func
current_pose = get_2d_pose(robot)
target_pose = get_2d_pose(goal)

for i in range(5):
  pr.step()
  depth = vision_sensor.capture_depth()
  occ_grid.from_depth(depth)

'''
dij_solver = Dijkstra(occ_grid, current_pose, target_pose)
moves = dij_solver.solve()
for i in moves:
  pr.step()
  set_2d_pose(robot, (i[0], i[1], current_pose[2]))
  time.sleep(0.05)
'''


# RRT Solver
rrt_solver = RRT(pr, robot, current_pose, target_pose, [-0.5,0.5], [-0.5,0.5])
moves = rrt_solver.solve()

for i in moves:
  pr.step()
  set_2d_pose(robot, (i[0], i[1], i[2]))
  time.sleep(0.05)

'''
coverage_planner = basicCC(occ_grid, current_pose, target_pose)

moves = coverage_planner.solve()
'''

for i in moves:
  pr.step()
  set_2d_pose(robot, (i[0], i[1], current_pose[2]))
  time.sleep(0.01)

# Get start state
# Get goal state

# End Simulation
pr.stop()
pr.shutdown()