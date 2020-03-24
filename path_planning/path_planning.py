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
SCENE_FILE = join(dirname(abspath(__file__)), '../scenes/scene_cpp.ttt')

# Set numpy printing options
np.set_printoptions(threshold=(100*100), formatter={'all':lambda x: str(x) + ','})

# Start Simulation
pr = PyRep()
pr.launch(SCENE_FILE, headless=True)
pr.start()
robot = Shape('robot')
goal = Shape('goal_pose')

robot.set_renderable(False)
goal.set_renderable(False)

vision_sensor = VisionSensor('vision_sensor')

# Since our "robot" is actually a shape we need to define our own 2d pose func
# current_pose = get_2d_pose(robot)
target_pose = get_2d_pose(goal)

current_pose =  robot.get_position(vision_sensor)[:2] + robot.get_orientation()[-1:]
current_pose = (-current_pose[0], current_pose[1], current_pose[2])

print(current_pose)
print(get_2d_pose(robot))

pr.step()
origin_x, _, origin_y, _, _, _ = vision_sensor.get_bounding_box()

# Setup occ_grid
occ_grid = OccupancyGrid(origin_x, origin_y)

depth = vision_sensor.capture_depth()
occ_grid.fromDepth(depth)

'''
dij_solver = Dijkstra(occ_grid, current_pose, target_pose)
moves = dij_solver.solve()
for i in moves:
  pr.step()
  set_2d_pose(robot, (i[0], i[1], current_pose[2]))
  time.sleep(0.05)
'''

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
# print(coverage_planner.dist_grid)


# for i in moves:
#   pr.step()
#   set_2d_pose(robot, (i[0], i[1], current_pose[2]))
#   time.sleep(0.1)

# Get start state
# Get goal state

# End Simulation
pr.stop()
pr.shutdown()