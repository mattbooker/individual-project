from enum import Enum
from math import sqrt
import numpy as np
import cv2

def raytrace(x0, y0, x1, y1):

  # Uses bresenham2d for raytracing

  line = []

  dx = abs(x1-x0)
  sx = 1 if x0 < x1 else -1
  dy = -abs(y1-y0)
  sy = 1 if y0 < y1 else -1
  err = dx+dy

  while (True):
    line.append((x0, y0))

    if (x0==x1 and y0==y1):
      break

    e2 = 2*err

    if (e2 >= dy):
      err += dy
      x0 += sx
    if (e2 <= dx):
      err += dx
      y0 += sy

  return line

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
  
  def nhood8(self):
    return [self + Point(-1, 0), self + Point(0, -1), self + Point(1, 0), self + Point(0, 1),
            self + Point(-1, -1), self + Point(-1, 1), self + Point(1, -1), self + Point(1, 1)]

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
  def __init__(self, corners : [Point], is_rectangle= True):
    # Clockwise order of corners
    self.corners = corners
    self.is_rectangle = is_rectangle

    # Get max and min values
    self.min_x = min(self.corners, key=lambda p: p.x).x
    self.min_y = min(self.corners, key=lambda p: p.y).y

    self.max_x = max(self.corners, key=lambda p: p.x).x
    self.max_y = max(self.corners, key=lambda p: p.y).y
    
    self.size_x = self.max_x - self.min_x + 1
    self.size_y = self.max_y - self.min_y + 1

    if is_rectangle:
      # Rounds down
      self.centre_x = (self.max_x + self.min_x)//2
      self.centre_y = (self.max_y + self.min_y)//2
      
    else:
      all_points = self.range()
      total_x = 0
      total_y = 0

      for x,y in all_points:
        total_x += x
        total_y += y

      self.centre_x = total_x//len(all_points)
      self.centre_y = total_y//len(all_points)
    

    # Sweep directions
    self.overall_direction = None
    self.initial_direction = None

  def range(self):

    values = set()

    if self.is_rectangle:
      # Simple iteration over bounds
      for j in range(self.min_y, self.max_y + 1):
        for i in range(self.min_x, self.max_x + 1):
          values.add((i,j))
    else:
      edge_points = np.array([[p.x, p.y] for p in self.corners])

      grid = np.zeros((100,100))
      cv2.fillPoly(grid, pts=[edge_points], color=[1])

      indices = np.transpose(np.nonzero(grid))

      for idx in indices:
        values.add((idx[1], idx[0]))

    return list(values)
