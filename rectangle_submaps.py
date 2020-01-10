from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
import time
import math

from maximumIndependentSet import MaximumIndependentSet
from collections import defaultdict
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

  def shift(self, direction):
    if direction == Direction.UP:
      return Point(self.x, self.y - 1)

    elif direction == Direction.DOWN:
      return Point(self.x, self.y + 1)

    elif direction == Direction.LEFT:
      return Point(self.x - 1, self.y)

    elif direction == Direction.RIGHT:
      return Point(self.x + 1, self.y)

class Direction(Enum):
  # Used for rectangular submaps
  UP = -1
  DOWN = -2
  LEFT = -3
  RIGHT = -4

  # Special is used for an edge case involving corners
  SPECIAL = -5
  
  # Intersection is used for when two marked edges intersect
  INTERSECTION = -6

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
      return self

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
      return self


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

def seperateCorners(corners, occ_grid):
  '''
  Returns all pairs of points in corners that are cogrid, i.e. that lie along the same x or y line. The result
  is split into two lists vertical pairs and horizontal pairs.
  '''

  # Naive O(n^2) implementation
  # Could be improved via sorting
  has_cogrid_pair = [False] * len(corners)

  vertical_pairs = []
  horizontal_pairs = []

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
          vertical_pairs.append((corners[a], corners[b]))
          has_cogrid_pair[a] = True
          has_cogrid_pair[b] = True
        
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
          horizontal_pairs.append((corners[a], corners[b]))

          has_cogrid_pair[a] = True
          has_cogrid_pair[b] = True

  
  remaining_corners = [concave_corners[i] for i in range(len(concave_corners)) if has_cogrid_pair[i] == False]

  return vertical_pairs, horizontal_pairs, remaining_corners

def getEdgeDirection(x, y, occ_grid):

  # [top, left, bottom, right]
  occupied = getNhoodOccupancy(x, y, occ_grid)

  # Take first free direction rather than random dir
  direction = occupied.index(False)

  # Check what to mark the cell as based on whats around it
  check_idx = (direction - 1) % 4

  if occupied[check_idx]:
    if check_idx == 0:
      return Direction.LEFT
      
    elif check_idx == 1:
      return Direction.DOWN

    elif check_idx == 3:
      return Direction.UP

    else:
      # The case where check_idx is 2 should never occur because we always choose the first unoccupied in CCW direction
      print("ERROR")
      exit() # TODO:

  # An edge case occurs when the cells above the corner and to the right of the corner are unoccupied
  elif occupied[(direction + 1) % 4]:
    return Direction.SPECIAL

def markEdge(x, y, direction, occ_grid):

  cur_point = Point(x, y)

  if direction == Direction.SPECIAL:

    # Move upwards
    direction = Direction.UP
    cur_point = cur_point.shift(direction)

    # Occurs if another edge runs over this one
    if occ_grid[cur_point.x, cur_point.y] < 0:
      return False

    while occ_grid.inBounds(cur_point.x, cur_point.y) and occ_grid[cur_point.x, cur_point.y] == 0:
      # But mark the cell with down
      occ_grid[cur_point.x, cur_point.y] = direction.opposite().value
      cur_point = cur_point.shift(direction)

    # TODO:
    if occ_grid.inBounds(cur_point.x, cur_point.y) and occ_grid[cur_point.x, cur_point.y] == direction.next().value:
      occ_grid[cur_point.x, cur_point.y] = Direction.INTERSECTION.value

  else:
    cur_point = cur_point.shift(direction)

    # Occurs if another edge runs over this one
    if occ_grid[cur_point.x, cur_point.y] < 0:
      return False

    while occ_grid.inBounds(cur_point.x, cur_point.y) and occ_grid[cur_point.x, cur_point.y] == 0:
      occ_grid[cur_point.x, cur_point.y] = direction.value
      cur_point = cur_point.shift(direction)

    # TODO:
    if occ_grid.inBounds(cur_point.x, cur_point.y) and occ_grid[cur_point.x, cur_point.y] == direction.next().value:
      occ_grid[cur_point.x, cur_point.y] = Direction.INTERSECTION.value

  return True

def markCogrid(x1, y1, x2, y2, vertical_line, occ_grid):
  occupied_1 = getNhoodOccupancy(x1, y1, occ_grid)
  occupied_2 = getNhoodOccupancy(x2, y2, occ_grid)

  # Right rotate occupied_1
  right_rotate = occupied_1[-1:] + occupied_1[:-1]

  # Left rotate occupied_1
  left_rotate = occupied_1[1:] + occupied_1[:1]

  if right_rotate == occupied_2:
    if vertical_line:
      # Start from top most point and move down
      start = Point(x1, min(y1, y2))
      markEdge(start.x, start.y, Direction.DOWN, occ_grid)

      # Return the start point shifted one (so that it is the corner of the rectangle) as well as an adjacent tile (that forms a potential rectangle)
      rect_corner = start.shift(Direction.DOWN)
      return [rect_corner], [rect_corner.shift(Direction.RIGHT)]

    else:
      # Start from left most point and move right
      start = Point(min(x1, x2), y1)
      markEdge(start.x, start.y, Direction.RIGHT, occ_grid)

      rect_corner = start.shift(Direction.RIGHT)
      return [rect_corner], [rect_corner.shift(Direction.UP)]
  
  elif left_rotate == occupied_2:
    if vertical_line:
      # Start from bottom most point and move up
      start = Point(x1, max(y1, y2))
      markEdge(start.x, start.y, Direction.UP, occ_grid)

      rect_corner = start.shift(Direction.UP)
      return [rect_corner], [rect_corner.shift(Direction.LEFT)]

    else:
      # Start from right most point and move left
      start = Point(max(x1, x2), y1)
      markEdge(start.x, start.y, Direction.LEFT, occ_grid)


      rect_corner = start.shift(Direction.LEFT)
      return [rect_corner], [rect_corner.shift(Direction.DOWN)]

  
  elif occupied_1 == occupied_2[::-1]:
    '''
    If we reach here it means we have two corners whose leading edge move in opposite directions i.e.
        |                     |
      --+     +--   or      --+
              |                     
                              +--
                              |
    '''

    if vertical_line:
      downward_start_point = Point(x1 - 1, min(y1,y2))
      upward_start_point = Point(x1 + 1, max(y1, y2))

      markEdge(downward_start_point.x, downward_start_point.y, Direction.DOWN, occ_grid)
      markEdge(upward_start_point.x, upward_start_point.y, Direction.UP, occ_grid)

      # TODO: Include the 1 line that forms between rectangles in possible_rect?
      return [downward_start_point.shift(Direction.DOWN), upward_start_point.shift(Direction.UP)], []

    else:      
      # For this case with horizontal lines, the edge ends on the corner rather than beginning at it thus we need to run markEdge backwards
      leftward_end_point = Point(min(x1, x2), y1 - 1)
      rightward_end_point = Point(max(x1, x2), y1 + 1)

      # Get the start points of the edge (using the end_point will double add a corner during the rectangle phase)
      result = []

      cur_point = leftward_end_point.shift(Direction.RIGHT)
      while occ_grid.inBounds(cur_point.x, cur_point.y) and occ_grid[cur_point.x, cur_point.y] == 0:
        occ_grid[cur_point.x, cur_point.y] = Direction.LEFT.value
        cur_point = cur_point.shift(Direction.RIGHT)

      result.append(cur_point.shift(Direction.LEFT))

      cur_point = rightward_end_point.shift(Direction.LEFT)
      while occ_grid.inBounds(cur_point.x, cur_point.y) and occ_grid[cur_point.x, cur_point.y] == 0:
        occ_grid[cur_point.x, cur_point.y] = Direction.RIGHT.value
        cur_point = cur_point.shift(Direction.LEFT)

      result.append(cur_point.shift(Direction.RIGHT))

      return result, []

  else:
    '''
    Same case as above except the edges are flipped i.e.
              |             |
      --+     +--   or      +--
        |                          
                          --+
                            |
    '''

    if vertical_line:
      # For this case with vertical lines, the edge ends on the corner rather than beginning at it thus we need to run the mark edge backwards
      upward_end_point = Point(x1 + 1, min(y1, y2))
      downward_end_point = Point(x1 - 1, max(y1, y2))

      # Get the start points of the edge (using the end_point will double add a corner during the rectangle phase)
      result = []

      cur_point = upward_end_point.shift(Direction.DOWN)
      while occ_grid.inBounds(cur_point.x, cur_point.y) and occ_grid[cur_point.x, cur_point.y] == 0:
        occ_grid[cur_point.x, cur_point.y] = Direction.UP.value
        cur_point = cur_point.shift(Direction.DOWN)

      result.append(cur_point.shift(Direction.UP))

      cur_point = downward_end_point.shift(Direction.UP)
      while occ_grid.inBounds(cur_point.x, cur_point.y) and occ_grid[cur_point.x, cur_point.y] == 0:
        occ_grid[cur_point.x, cur_point.y] = Direction.DOWN.value
        cur_point = cur_point.shift(Direction.UP)

      result.append(cur_point.shift(Direction.DOWN))

      return result, []

    else:
      leftward_initial_point = Point(max(x1, x2), y1 - 1)
      rightward_initial_point = Point(min(x1, x2), y1 + 1)

      markEdge(leftward_initial_point.x, leftward_initial_point.y, Direction.LEFT, occ_grid)
      markEdge(rightward_initial_point.x, rightward_initial_point.y, Direction.RIGHT, occ_grid)

      return [leftward_initial_point.shift(Direction.LEFT), rightward_initial_point.shift(Direction.RIGHT)], []

def makeRectangle(x, y, possible_rectangles, occ_grid):
  '''

  There is an edge case that occurs when an UP edge is broken in two by a LEFT edge. However this is a non-issue since the 
  order in which we search for edges and mark them is from top to bottom from left to right. This means the LEFT edge will always
  occur first before an upwards edge, i.e. only a LEFT edge will be split by an UP edge not vice-versa.
  '''

  initial_point = Point(x,y)

  cur_dir = Direction(occ_grid[x, y])
  cur_point = initial_point
  cur_point = cur_point.shift(cur_dir)

  rectangle_corners = [initial_point]

  cell_value = occ_grid[cur_point.x, cur_point.y]

  # Check the edge case where we start on a special corner
  if cur_dir == Direction.DOWN and occ_grid[cur_point.x, cur_point.y] == 1:
    cur_dir = Direction.LEFT
    cur_point = initial_point
    cur_point = cur_point.shift(cur_dir)

  while cur_point != initial_point:

    # Update the possible rectangles
    if cur_point in possible_rectangles:
      possible_rectangles.remove(cur_point)

    if len(rectangle_corners) > 4:
      print("PROBLEM")
      print(rectangle_corners)
      exit()

    # Check that the current point is in bounds
    if occ_grid.inBounds(cur_point.x, cur_point.y):
      cell_value = occ_grid[cur_point.x, cur_point.y]

    # Otherwise backtrack, change direction and mark the point as a corner in the rectangle
    else:
      cur_point = cur_point.shift(cur_dir.opposite())
      cur_dir = cur_dir.next()
      rectangle_corners.append(cur_point)

      # Move in new direction so that we dont have issues with the error checking steps below
      cur_point = cur_point.shift(cur_dir)

      # Go to next iteration
      continue
    
    # print(cur_point, end="--- ")
    # print(cell_value)

    # If we hit a cell that has 0 or the current direction then we continue moving in same direction
    if cell_value == 0 or cell_value == cur_dir.value:
      cur_point = cur_point.shift(cur_dir)

    # If we encounter an a cell with the next direction then add the corner and follow the next direction
    elif cell_value == cur_dir.next().value:
      # Move with new direction
      cur_dir = cur_dir.next()
      rectangle_corners.append(cur_point)

      cur_point = cur_point.shift(cur_dir)

    # If we encounter an intersection check the value in the next direction after the intersection.
    elif cell_value == Direction.INTERSECTION.value:
      cur_dir = cur_dir.next()
      potential_corner = cur_point
      cur_point = cur_point.shift(cur_dir)

      if occ_grid[cur_point.x, cur_point.y] == cur_dir.value:
        rectangle_corners.append(potential_corner)

      else:
        potential_corner = potential_corner.shift(cur_dir.next())
        rectangle_corners.append(potential_corner)
        cur_point = potential_corner.shift(cur_dir)


    # If we hit an obstacle (i.e. 1) or other marked cell then backtrack, change direction and mark point as a corner in the rectangle
    else:
      cur_point = cur_point.shift(cur_dir.opposite())
      cur_dir = cur_dir.next()

      rectangle_corners.append(cur_point)

      # Move in new direction so that we dont have issues with the error checking steps
      cur_point = cur_point.shift(cur_dir)
  
  return Submap(rectangle_corners)

def splitIntoRectangles(concave_corners, occ_grid):
  '''
  Given a list of concave corner points, splits the occ_grid into rectangles. Returns a list of points that are the corners of unique rectangles and
  a set of points that lie on the edges of other potential rectangles that were missed. For any split it can create either 1 or 2 rectangles, thus
  the set is used to keep track of extra rectangles.
  '''
  
  definite_rectangles = []
  possible_rectangles = set()

  for corner in concave_corners:
    direction = getEdgeDirection(corner.x, corner.y, occ_grid)
    suitable_edge = markEdge(corner.x, corner.y, direction, occ_grid)

    if not suitable_edge:
      continue

    # Based on the direction of the edge add the corner of the rectangle as well as the adjacent rectangle
    if direction == Direction.SPECIAL:
      definite_rectangles.append(corner.shift(Direction.UP))

      # Offset contains the previous shift
      possible_rectangles.add(corner.shift(Direction.RIGHT))
    else:
      definite_rectangles.append(corner.shift(direction))

      # Offset contains the previous shift
      possible_rectangles.add(corner.shift(direction.next().opposite()))

  return definite_rectangles, possible_rectangles

    
def extractSubmaps(definite_rectangles, possible_rectangles, occ_grid):
  submaps = []
  used_corners = set()

  # STEP 1: Iterate through the points known to be rectangle corners
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
        cell = cell.shift(direction_of_corner)
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
    
    # if direction_of_corner == None:
    #   print(occupied)

    # Set the cell to have the correct direction and then make a submap
    occ_grid[cell.x, cell.y] = direction_of_corner.value
    submaps.append(makeRectangle(cell.x, cell.y, possible_rectangles, occ_grid))

  # Visualization
  for num, submap in enumerate(submaps):
    for (x, y) in submap.range():
      occ_grid[x, y] = num + 10

  return submaps

# Get Scene
SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_cogrid_aligned.ttt')

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

# TODO: Manage the case where points are off by one
# TODO: Manage the case where points are on same line
vertical_cogrid, horizontal_cogrid, noncogrid_corners = seperateCorners(concave_corners, occ_grid)

bipartite_graph = defaultdict(list)
isolated_vertical = []
isolated_horizontal = []

# Create a temp_occ_grid to use for faster construction of the bipartite graph (if we use occ_grid it will leave remnants which we dont want)
temp_occ_grid = occ_grid.clone()

# Mark the vertical cogrid lines
for num, (point_1, point_2) in enumerate(vertical_cogrid, 2):

  # Find start and end points
  start_y = min(point_1.y, point_2.y)
  end_y = max(point_1.y, point_2.y)

  # Mark the cells from start to end with num
  for i in range(start_y, end_y + 1):
    temp_occ_grid[point_1.x, i] = num
  
# Mark horizontal cogrid lines and build the bipartite graph
for num, (point_1, point_2) in enumerate(horizontal_cogrid, 2 + len(vertical_cogrid)):
  
  independent_node = True

  # Find start and end poitns
  start_x = min(point_1.x, point_2.x)
  end_x = max(point_1.x, point_2.x)

  for i in range(start_x, end_x + 1):
    val = int(temp_occ_grid[i, point_1.y])

    if val > 1:
      independent_node = False
      bipartite_graph[val].append(num)

    temp_occ_grid[i, point_1.y] = num

  # Keep track of the point if it is not part of the graph
  if independent_node:
    isolated_horizontal.append((point_1, point_2))

for num, pair in enumerate(vertical_cogrid, 2):
  if num not in bipartite_graph:
    isolated_vertical.append(pair)



MIS = MaximumIndependentSet(bipartite_graph)
MIS.compute()
# print(bipartite_graph)

# print(vertical_cogrid)
# print(horizontal_cogrid)
# print(MIS.pairs)
# print(MIS.max_independent_set)
# print(MIS.min_vertex_cover)

cogrid_definite_rect = []
cogrid_possible_rect = []
used_points = set()

# Mark the corners that are part of the MIS
for num in MIS.max_independent_set:
  idx = num - 2

  rect_corner = None
  possible_rect = None

  if idx < len(vertical_cogrid):
    point_1, point_2 = vertical_cogrid[idx]
    rect_corner, possible_rect = markCogrid(point_1.x, point_1.y, point_2.x, point_2.y, True, occ_grid)

    used_points.add(point_1)
    used_points.add(point_2)

  else:
    idx -= len(vertical_cogrid)
    point_1, point_2 = horizontal_cogrid[idx]
    rect_corner, possible_rect = markCogrid(point_1.x, point_1.y, point_2.x, point_2.y, False, occ_grid)

    used_points.add(point_1)
    used_points.add(point_2)

  cogrid_definite_rect.extend(rect_corner)
  cogrid_possible_rect.extend(possible_rect)

for num in MIS.min_vertex_cover:
  idx = num - 2

  if idx < len(vertical_cogrid):
    point_1, point_2 = vertical_cogrid[idx]
    
    if point_1 not in used_points:
      noncogrid_corners.append(point_1)
    
    if point_2 not in used_points:
      noncogrid_corners.append(point_2)

  else:
    idx -= len(vertical_cogrid)
    point_1, point_2 = horizontal_cogrid[idx]

    if point_1 not in used_points:
      noncogrid_corners.append(point_1)
    
    if point_2 not in used_points:
      noncogrid_corners.append(point_2)

# Mark cogrid corners that were not included in the bipartite graph
for (point_1, point_2) in isolated_vertical:
  rect_corner, possible_rect = markCogrid(point_1.x, point_1.y, point_2.x, point_2.y, True, occ_grid)

  cogrid_definite_rect.extend(rect_corner)
  cogrid_possible_rect.extend(possible_rect)

for (point_1, point_2) in isolated_horizontal:
  rect_corner, possible_rect = markCogrid(point_1.x, point_1.y, point_2.x, point_2.y, False, occ_grid)
  
  cogrid_definite_rect.extend(rect_corner)
  cogrid_possible_rect.extend(possible_rect)


definite_rectangles, possible_rectangles = splitIntoRectangles(noncogrid_corners, occ_grid)

definite_rectangles.extend(cogrid_definite_rect)
possible_rectangles.update(cogrid_possible_rect)

submaps = extractSubmaps(definite_rectangles, possible_rectangles, occ_grid)
print(occ_grid)
# print(submaps[0].corners)
# print(occ_grid)

# End Simulation
pr.stop()
pr.shutdown()