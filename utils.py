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

  def distance_to(self, rhs):
    return sqrt((self.x + rhs.x)**2 + (self.y + rhs.y)**2)

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
  def __init__(self, corners : [Point]):
    # Clockwise order of corners
    self.corners = corners

    self.size_x = abs(corners[0].x - corners[2].x)
    self.size_y = abs(corners[0].y - corners[2].y)
    
    # Give mins and maxs initial values
    self.min_x = corners[0].x
    self.min_y = corners[0].y

    self.max_x = corners[0].x
    self.max_y = corners[0].y

    # Actually find the min and max values
    for p in corners:
      if p.x < self.min_x:
        self.min_x = p.x
      if p.y < self.min_y:
        self.min_y = p.y

      if p.x > self.max_x:
        self.max_x = p.x
      if p.y > self.max_y:
        self.max_y = p.y


    # Rounds down
    self.centre_x = (self.max_x + self.min_x)//2
    self.centre_y = (self.max_y + self.min_y)//2

  def range(self):
    min_x = min(self.corners, key=lambda p: p.x).x
    min_y = min(self.corners, key=lambda p: p.y).y

    max_x = max(self.corners, key=lambda p: p.x).x
    max_y = max(self.corners, key=lambda p: p.y).y


    for j in range(min_y, max_y + 1):
      for i in range(min_x, max_x + 1):
        yield (i,j)