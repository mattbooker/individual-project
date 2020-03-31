from enum import Enum
from math import sqrt

class Pose:
  def __init__(self, x, y, theta):
    self.x = x
    self.y = y
    self.theta = theta

  def __str__(self):
    return '{}, {}, {}'.format(self.x, self.y, self.theta)

  def __add__(self, rhs):
    return Pose(self.x + rhs.x, self.y + rhs.y, self.theta + rhs.theta)

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def distanceTo(self, rhs):
    return sqrt((self.x - rhs.x)**2 + (self.y - rhs.y)**2)

  def nhood4(self):
    return [self + Point(-1, 0), self + Point(0, -1), self + Point(1, 0), self + Point(0, 1)]

  def copy(self):
    return Point(self.x, self.y)

  def __str__(self):
    return '{}, {}'.format(self.x, self.y)

  def __add__(self, rhs):
    return Point(self.x + rhs.x, self.y + rhs.y)

  def __eq__(self, rhs):
    return self.x == rhs.x and self.y == rhs.y

  def __lt__(self, rhs):
    return self.x < rhs.x

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
  
  def isHorizontal(self):
    if self.name == "LEFT" or self.name == "RIGHT":
      return True
    else:
      return False

class Submap:
  def __init__(self, corners : [Point]):
    # Clockwise order of corners
    self.corners = corners

    self.size_x = abs(corners[0].x - corners[2].x)
    self.size_y = abs(corners[0].y - corners[2].y)
    
    # Get max and min values
    self.min_x = min(self.corners, key=lambda p: p.x).x
    self.min_y = min(self.corners, key=lambda p: p.y).y

    self.max_x = max(self.corners, key=lambda p: p.x).x
    self.max_y = max(self.corners, key=lambda p: p.y).y

    # Rounds down
    self.centre_x = (self.max_x + self.min_x)//2
    self.centre_y = (self.max_y + self.min_y)//2

    # Sweep directions
    self.overall_direction = None
    self.initial_direction = None

  def range(self):
    for j in range(self.min_y, self.max_y + 1):
      for i in range(self.min_x, self.max_x + 1):
        yield (i,j)