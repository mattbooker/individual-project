from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
import time
import math

from collections import namedtuple
from occupancyGrid import OccupancyGrid
from enum import Enum

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __str__(self):
    return '{}, {}'.format(self.x, self.y)

  def __add__(self, rhs):
    return Point(self.x + rhs.x, self.y + rhs.y)

  def __eq__(self, rhs):
    return self.x == rhs.x and self.y == rhs.y

  def __hash__(self):
    return self.x * 1000 + self.y

  def __repr__(self):
    return 'Point({})'.format(str(self))

class Direction(Enum):
  # Used for rectangular submaps
  UP = -1
  DOWN = -2
  LEFT = -3
  RIGHT = -4
  SPECIAL = -5

  def next(self):
    if self.name == "UP":
      return Direction.RIGHT
    elif self.name == "DOWN":
      return Direction.LEFT
    elif self.name == "LEFT":
      return Direction.UP
    elif self.name == "RIGHT":
      return Direction.DOWN
    else:
      return Direction.SPECIAL

  def opposite(self):
    if self.name == "UP":
      return Direction.DOWN
    elif self.name == "DOWN":
      return Direction.UP
    elif self.name == "LEFT":
      return Direction.RIGHT
    elif self.name == "RIGHT":
      return Direction.LEFT
    else:
      return Direction.SPECIAL

class Submap:
  def __init__(self, corners):
    # Clockwise order of corners
    self.corners = corners


    # self.centre_x = (x1 + x2)/2
    # self.centre_y = (y1 + y2)/2
    self.size_x = abs(corners[0].x - corners[2].x)
    self.size_y = abs(corners[0].y - corners[2].y)

  def range(self):
    min_x = min(self.corners, key=lambda p: p.x).x
    min_y = min(self.corners, key=lambda p: p.y).y

    max_x = max(self.corners, key=lambda p: p.x).x
    max_y = max(self.corners, key=lambda p: p.y).y


    for j in range(min_y, max_y + 1):
      for i in range(min_x, max_x + 1):
        yield (i,j)
    

def setupOccGrid(occ_grid, vision_sensor):
  # Capture vision depth and create occupancy grid
  pr.step()
  depth = vision_sensor.capture_depth()
  occ_grid.fromDepth(depth)

def getNhoodOccupancy(x, y, occ_grid):
  '''
  Returns a bool array with 4 elements corresponding to whether the top, left, bottom or right cells are occupied.
  Out of bounds is treated as unoccupied.
  '''

  # Top, Left, Bottom, Right
  occupied = [False, False, False, False]

  # Check Top
  if y - 1 >= 0 and occ_grid[x, y - 1] == 1:
    occupied[0] = True

  # Check Left
  if x - 1 >= 0 and occ_grid[x - 1, y] == 1:
    occupied[1] = True

  # Check Bottom
  if y + 1 < occ_grid.size_y and occ_grid[x, y + 1] == 1:
    occupied[2] = True

  # Check Right
  if x + 1 < occ_grid.size_x and occ_grid[x + 1, y] == 1:
    occupied[3] = True

  return occupied

def getNhoodValues(x, y, occ_grid):
  '''
  Returns an array containing the values above, left, below and right of the given x,y.
  Out of bounds is treated as 1.
  '''

  # Top, Left, Bottom, Right
  occupied = [1, 1, 1, 1]

  # Check Top
  if y - 1 >= 0:
    occupied[0] = occ_grid[x, y - 1]

  # Check Left
  if x - 1 >= 0:
    occupied[1] = occ_grid[x - 1, y]

  # Check Bottom
  if y + 1 < occ_grid.size_y:
    occupied[2] = occ_grid[x, y + 1]

  # Check Right
  if x + 1 < occ_grid.size_x:
    occupied[3] = occ_grid[x + 1, y]

  return occupied

def isCorner(x, y, occ_grid):
  '''
  A corner is defined to be when a 3x3 grid has its centre unoccupied and ...
  '''

  '''
  0 | 1 | 2
  7 |   | 3 
  6 | 5 | 4

  Marks the non-occupied cell diagonal to the corner

  if occ_grid[x, y] != 0:
    return False
  
  offsets = [Point(-1,-1), Point(0,-1), Point(1,-1), Point(1,0), Point(1,1), Point(0,1), Point(-1,1), Point(-1,0)]
  current_point = Point(x,y)
  occupied = [False] * 8

  for num, o in enumerate(offsets):
    point = current_point + o

    # If the point is out of bounds then mark as true
    if point.x < 0 or point.y < 0 or point.x >= occ_grid.size_x or point.y >= occ_grid.size_y:
      occupied[num] = True
    elif occ_grid[point.x, point.y] == 1:
      occupied[num] = True

  # If one is occupied and its the corners then return true
  if sum(occupied) == 1:
    if occupied[0] or occupied[2] or occupied[4] or occupied[6]:
      return True

  # If 3 are occupied then check to ensure they are on the corners and the opposite corner has no occupied cells
  elif sum(occupied) >= 3:
    if (occupied[7] and occupied[0] and occupied[1]) and not (occupied[3] and occupied[4] and occupied[5]):
      return True
    elif (occupied[1] and occupied[2]   and occupied[3]) and not (occupied[5] and occupied[6] and occupied[7]):
      return True
    elif (occupied[3] and occupied[4] and occupied[5]) and not (occupied[7] and occupied[0] and occupied[1]):
      return True
    elif (occupied[5] and occupied[6] and occupied[7]) and not (occupied[1] and occupied[2] and occupied[3]):
      return True

  return False
  '''

  # If point is not occupied then it cant be a corner
  if occ_grid[x, y] == 0:
    return False

  on_boundary = False

  occupied = getNhoodOccupancy(x, y, occ_grid)

  if x == 0 or x == occ_grid.size_x - 1 or y == 0 or y == occ_grid.size_y - 1:
    on_boundary = True

  # If we are on the boundary then a corner only forms if a surrounding 4-neighbourhood has 1 occupied
  if on_boundary:
    if sum(occupied) == 1:
      return True
  elif sum(occupied) == 2:  # Otherwise a corner only forms when the 4-nhood has 2 occupied
    # Check the case where the two occupied positions form a straight line
    if (occupied[0] and occupied[2]) or (occupied[1] and occupied[3]):
      return False
    else:
      return True

  return False

def getCorners(occ_grid):
  corners = []

  for j in range(occ_grid.size_y):
    for i in range(occ_grid.size_x):
      if isCorner(i, j, occ_grid):
        corners.append(Point(i,j))
  
  return corners

def getCogridCorners(corners, occ_grid):
  # Naive O(n^2) implementation
  # Could be improved via sorting

  pairs = []
  unique_points = set()

  # Iterate through all points to find cogrid vertices. Need to ensure a clear line exists between the two points
  for a in range(len(corners) - 1):
    for b in range(a + 1, len(corners)):
      if corners[a].x == corners[b].x: 

        # Check there is an uniterrupted line between the two points
        start = min(corners[a].y, corners[b].y)
        end = max(corners[a].y, corners[b].y)

        isValid = True

        for idx in range(start + 1, end):
          if occ_grid[corners[a].x, idx] == 1:
            isValid = False
            break
        
        # If the two points form a valid cogrid pair then add to our result
        if isValid:
          pairs.append((corners[a], corners[b]))

          unique_points.add(corners[a])
          unique_points.add(corners[b])
        
      elif corners[a].y == corners[b].y:

        # Check there is an uniterrupted line between the two points
        start = min(corners[a].x, corners[b].x)
        end = max(corners[a].x, corners[b].x)

        isValid = True

        for idx in range(start + 1, end):
          if occ_grid[idx, corners[a].y] == 1:
            isValid = False
            break

        # If the two points form a valid cogrid pair then add to our result
        if isValid:
          pairs.append((corners[a], corners[b]))

          unique_points.add(corners[a])
          unique_points.add(corners[b])

  return pairs, len(unique_points)

def markEdge(x, y, direction, occ_grid):
  if direction == Direction.UP:
    while y >= 0 and occ_grid[x, y] == 0:
      occ_grid[x, y] = Direction.UP.value
      y -= 1
  
  elif direction == Direction.DOWN:
    while y < occ_grid.size_y and occ_grid[x, y] == 0:
      occ_grid[x, y] = Direction.DOWN.value
      y += 1

  elif direction == Direction.LEFT:
    while x >= 0 and occ_grid[x, y] == 0:
      occ_grid[x, y] = Direction.LEFT.value
      x -= 1

  elif direction == Direction.RIGHT:
    while x < occ_grid.size_x and occ_grid[x, y] == 0:
      occ_grid[x, y] = Direction.RIGHT.value
      x += 1

  elif direction == Direction.SPECIAL:
    while y >= 0 and occ_grid[x, y] == 0:
      occ_grid[x, y] = Direction.DOWN.value
      y -= 1

def makeRectangle(x, y, possible_rectangles, occ_grid):
  '''

  There is an edge case that occurs when an UP edge is broken in two by a LEFT edge. However this is a non-issue since the 
  order in which we search for edges and mark them is from top to bottom from left to right. This means the LEFT edge will always
  occur first before an upwards edge, i.e. only a LEFT edge will be split by an UP edge not vice-versa.
  '''

  shift = {
      Direction.UP : Point(0, -1), 
      Direction.DOWN : Point(0, 1), 
      Direction.LEFT : Point(-1, 0), 
      Direction.RIGHT : Point(1, 0)}
  
  initial_point = Point(x,y)

  cur_dir = Direction(occ_grid[x, y])
  cur_point = initial_point + shift[cur_dir]

  rectangle_corners = [initial_point]

  cell_value = occ_grid[cur_point.x, cur_point.y]

  # Check the edge case where we start on a special corner
  if cur_dir == Direction.DOWN and occ_grid[cur_point.x, cur_point.y] == 1:
    cur_dir = Direction.LEFT
    cur_point = initial_point + shift[cur_dir]

  while cur_point != initial_point:

    # Update the possible rectangles
    if cur_point in possible_rectangles:
      possible_rectangles.remove(cur_point)

    if len(rectangle_corners) > 4:
      print("PROBLEM")
      print(rectangle_corners)
      exit()

    # TODO: Check that the cells we go over arent part of potential rectangles

    # Check that the current point is in bounds
    if occ_grid.inBounds(cur_point.x, cur_point.y):
      cell_value = occ_grid[cur_point.x, cur_point.y]

    # Otherwise backtrack, change direction and mark the point as a corner in the rectangle
    else:
      cur_point += shift[cur_dir.opposite()]
      cur_dir = cur_dir.next()
      rectangle_corners.append(cur_point)

      # Move in new direction so that we dont have issues with the error checking steps below
      cur_point += shift[cur_dir]

      # Go to next iteration
      continue

    # print(cur_point, end="--- ")
    # print(cell_value)

    # If we hit a cell that has 0 or the current direction then we continue moving in same direction
    if cell_value == 0 or cell_value == cur_dir.value:
      cur_point += shift[cur_dir]

    # If we encounter a cell that tells us to change direction and is correct then follow the new direction
    elif cell_value == cur_dir.next().value:
      # Move with new direction
      cur_dir = cur_dir.next()
      rectangle_corners.append(cur_point)

    # If we hit an obstacle (i.e. 1) or other marked cell then backtrack, change direction and mark point as a corner in the rectangle
    else:
      cur_point += shift[cur_dir.opposite()]
      cur_dir = cur_dir.next()

      rectangle_corners.append(cur_point)

      # Move in new direction so that we dont have issues with the error checking steps
      cur_point += shift[cur_dir]
  

  return Submap(rectangle_corners)
    
def extractSubmaps(definite_rectangles, possible_rectangles, occ_grid):
  submaps = []
  used_corners = set()

  shift = {
      Direction.UP : Point(0, -1), 
      Direction.DOWN : Point(0, 1), 
      Direction.LEFT : Point(-1, 0), 
      Direction.RIGHT : Point(1, 0)}

  # STEP 1: Iterate through the known to be rectangles
  for num, corner in enumerate(definite_rectangles):

    # Skip this corner if it has been used in another rectangle
    if corner in used_corners:
      continue

    submap = makeRectangle(corner.x, corner.y, possible_rectangles, occ_grid)

    # Add the corners of this submap into the used corners set
    for p in submap.corners:
      used_corners.add(p)

    submaps.append(submap)


  # STEP 2: Iterate through the rectangles that may have not been accounted for
  while len(possible_rectangles) > 0:
    cell = possible_rectangles.pop()

    occupied = getNhoodValues(cell.x, cell.y, occ_grid)

    direction_of_corner = None

    if occupied.count(0) == 3:
      if occupied[0]:
        direction_of_corner = Direction.RIGHT
      elif occupied[1]:
        direction_of_corner = Direction.UP
      elif occupied[2]:
        direction_of_corner = Direction.LEFT
      elif occupied[3]:
        direction_of_corner = Direction.DOWN

      # Need to move to the closest corner
      while occupied.count(0) != 2:
        cell += shift[direction_of_corner]
        occupied = getNhoodValues(cell.x, cell.y, occ_grid)

      direction_of_corner = direction_of_corner.next()

    else:
      if occupied[3] and occupied[0]:
        direction_of_corner = Direction.DOWN
      elif occupied[0] and occupied[1]:
        direction_of_corner = Direction.RIGHT
      elif occupied[1] and occupied[2]:
        direction_of_corner = Direction.UP
      elif occupied[2] and occupied[3]:
        direction_of_corner = Direction.LEFT

    if direction_of_corner == None:
      print(occupied)

    # Set the cell to have the correct direction and then make a submap
    occ_grid[cell.x, cell.y] = direction_of_corner.value
    submaps.append(makeRectangle(cell.x, cell.y, possible_rectangles, occ_grid))

  # # Visualization
  # for num, submap in enumerate(submaps):
  #   for (x, y) in submap.range():
  #     occ_grid[x, y] = num + 10

  return submaps

# Get Scene
SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_cogrid_test.ttt')

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

pr.step()
concave_corners = getCorners(occ_grid)

cogrid_corners, number_of_unique_points = getCogridCorners(concave_corners, occ_grid)

# Create an adjacency matrix
cogrid_graph = np.zeros((number_of_unique_points, number_of_unique_points))

#TODO: Get Maximal independent set of cogrid

# Create a temp_occ_grid to use for the marking in finding the cogrid
temp_occ_grid = occ_grid.clone()

# Construct the graph
for num, (point_1, point_2) in enumerate(cogrid_corners, 2):

  line = []

  # x is constant
  if point_1.x == point_2.x:
    start_y = min(point_1.y, point_2.y)
    end_y = max(point_1.y, point_2.y)

    for i in range(start_y, end_y + 1):
      val = int(temp_occ_grid[point_1.x, i])

      if val > 1:
        # subtract 2 since we are using the interval [2,..] for the numbers
        cogrid_graph[val - 2, num - 2] = 1
        cogrid_graph[num - 2, val - 2] = 1

      temp_occ_grid[point_1.x, i] = num

  # y is constant
  else:
    start_x = min(point_1.x, point_2.x)
    end_x = max(point_1.x, point_2.x)

    for i in range(start_x, end_x + 1):
      val = int(temp_occ_grid[i, point_1.y])

      if val > 1:
        # subtract 2 since we are using the interval [2,..] for the numbers
        cogrid_graph[val - 2, num - 2] = 1
        cogrid_graph[num - 2, val - 2] = 1

      temp_occ_grid[i, point_1.y] = num


# print(temp_occ_grid)
print(cogrid_graph)
print(cogrid_corners)


# definite_rectangles = []
# possible_rectangles = set()

# # Mark the Rectangles
# for corner in concave_corners:

#   occupied = getNhoodOccupancy(corner.x, corner.y, occ_grid)

#   # Take first free direction rather than random dir
#   direction = occupied.index(False)

#   # Check what to mark the cell as based on whats around it
#   check_idx = (direction - 1) % 4

#   if occupied[check_idx]:
#     if check_idx == 0:
#       markEdge(corner.x - 1, corner.y, Direction.LEFT, occ_grid)

#       definite_rectangles.append(Point(corner.x - 1, corner.y))
#       possible_rectangles.add(Point(corner.x - 1, corner.y + 1))
      
#     elif check_idx == 1:
#       markEdge(corner.x, corner.y + 1, Direction.DOWN, occ_grid)

#       definite_rectangles.append(Point(corner.x, corner.y + 1))
#       possible_rectangles.add(Point(corner.x + 1, corner.y + 1))

#     elif check_idx == 3:
#       markEdge(corner.x, corner.y - 1, Direction.UP, occ_grid)

#       definite_rectangles.append(Point(corner.x, corner.y - 1))
#       possible_rectangles.add(Point(corner.x - 1, corner.y - 1))

#     else:
#       # The case where check_idx is 2 should never occur because we always choose the first unoccupied in CCW direction
#       print("ERROR")
#       exit() # TODO:

#   elif occupied[(direction + 1) % 4]:
#     markEdge(corner.x, corner.y - 1, Direction.SPECIAL, occ_grid)

#     definite_rectangles.append(Point(corner.x, corner.y - 1))
#     possible_rectangles.add(Point(corner.x + 1, corner.y - 1))

# submaps = extractSubmaps(definite_rectangles, possible_rectangles, occ_grid)

# print(occ_grid)

# End Simulation
pr.stop()
pr.shutdown()