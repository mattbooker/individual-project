import occupancyGrid
import heapq as h
import numpy as np
from collections import defaultdict, deque

import time
import matplotlib.pyplot as plt
from utils import Direction, Point

class basicCC:

  def __init__(self, occ_grid, start_pose, target_pose):
    self.occ_grid = occ_grid
    self.dist_grid = occ_grid.clone()
    # self.dist_grid.grid = np.full((self.dist_grid.size_x,self.dist_grid.size_y), 0)

    self.start_mx, self.start_my = occ_grid.worldToMap(start_pose[0], start_pose[1])
    self.goal_mx, self.goal_my = occ_grid.worldToMap(target_pose[0], target_pose[1])

  def nhood4(self, x, y):
    result = []

    if x >= 1 and self.occ_grid[x-1, y] == 0:
      result.append((x - 1, y))

    if x < self.occ_grid.size_x - 1 and self.occ_grid[x+1, y] == 0:
      result.append((x + 1, y))

    if y >= 1 and self.occ_grid[x, y-1] == 0:
      result.append((x, y - 1))

    if y < self.occ_grid.size_y - 1 and self.occ_grid[x, y+1] == 0:
      result.append((x, y + 1))

    return result

  def nhood8(self, x, y):
    result = []

    if x >= 1 and self.occ_grid[x-1, y] == 0:
      result.append((x - 1, y))

    if x < self.occ_grid.size_x - 1 and self.occ_grid[x+1, y] == 0:
      result.append((x + 1, y))

    if y >= 1 and self.occ_grid[x, y-1] == 0:
      result.append((x, y - 1))

    if y < self.occ_grid.size_y - 1 and self.occ_grid[x, y+1] == 0:
      result.append((x, y + 1))

    if x >= 1 and y >= 1 and self.occ_grid[x-1, y-1] == 0:
      result.append((x - 1, y - 1))

    if x >= 1 and y < self.occ_grid.size_y - 1 and self.occ_grid[x-1, y+1] == 0:
      result.append((x - 1, y + 1))

    if x < self.occ_grid.size_x - 1 and y >= 1 and self.occ_grid[x+1, y-1] == 0:
      result.append((x + 1, y - 1))

    if x < self.occ_grid.size_x - 1 and y < self.occ_grid.size_y - 1 and self.occ_grid[x+1, y+1] == 0:
      result.append((x + 1, y + 1))

    return result

  def solve(self):
    # self.compute_dist_grid()
    # print(self.dist_grid)

    # TODO: Parametrize the size
    footprint = Footprint(self.start_mx, self.start_my, 20, 2)
    print(self.start_mx, self.start_my)

    moves = footprint.lawnmower(self.occ_grid.size_x, self.occ_grid.size_y, Direction.LEFT, Direction.UP)
  
    for i in moves:
      footprint.move_to(i.x, i.y)
      footprint.fill_footprint(self.occ_grid, 1)
      plt.imshow(self.occ_grid.grid)
      plt.pause(0.001)
      plt.clf()
    # return None

  def get_path(self, footprint):
    start_point = (self.start_mx, self.start_my)

    result = [start_point]

    # DEBUGGING
    count = 0

    while True:
      count += 1
      footprint.fill_footprint(self.dist_grid)
      
      next_mx, next_my, rotation_required = footprint.get_next_movement(self.dist_grid)

      if next_mx == -1 and next_my == -1:
        return result
      else:
        result.append(self.occ_grid.mapToWorld(next_mx, next_my))

      footprint.move_to(next_mx, next_my, rotation_required)

      if count > 1:
        plt.imshow(self.dist_grid.grid)
        plt.pause(0.001)
        plt.clf()
        # plt.show()
        # break


    return result[::-1]

  def compute_dist_grid(self):
    '''
    A breadth-first search of the grid to populate with the distance to goal
    '''

    visited = set()
    q = deque()

    q.append((self.goal_mx, self.goal_my))

    # Costs start from 1
    self.dist_grid[self.goal_mx, self.goal_my] = 5

    count = 0

    while len(q) > 0:
      count += 1
      current_mx, current_my = q.popleft()

      for nx, ny in self.nhood8(current_mx, current_my):
        if (nx, ny) not in visited:
          visited.add((nx, ny))
          self.dist_grid[nx, ny] = self.dist_grid[current_mx, current_my] + 1
          q.append((nx, ny))

class Footprint:

  def __init__(self, current_x, current_y, size_x, size_y):
    self.cur_pos = Point(current_x, current_y)

    self.size_x = size_x
    self.size_y = size_y

    # Make the shape have an odd size -> leads to simpler calculations
    # if size_x % 2 == 0:
    #   self.size_x += 1
    
    # if size_y % 2 == 0:
    #   self.size_y += 1

  def compute_corners(self, pos, size_x, size_y):

    # Top left, top right, bottom right, bottom left
    # (Clockwise order from top left)
    corners = [Point(0,0)] * 4

    corners[0] = Point(int(pos.x - (size_x - 1)/2), int(pos.y - (size_y - 1)/2))
    corners[1] = Point(int(pos.x + (size_x - 1)/2), int(pos.y - (size_y - 1)/2))
    corners[2] = Point(int(pos.x + (size_x - 1)/2), int(pos.y + (size_y - 1)/2))
    corners[3] = Point(int(pos.x - (size_x - 1)/2), int(pos.y + (size_y - 1)/2))
    
    return corners

  def fill_footprint(self, grid, fill_val = 0):

    corners = self.compute_corners(self.cur_pos, self.size_x, self.size_y)

    for i in range(corners[0].x, corners[1].x + 1):
      for j in range(corners[0].y, corners[2].y + 1):

        assert grid.inBounds(i, j), "Footprint at %r, %r is out of bounds" % (i, j)

        grid[i, j] = fill_val

  def get_next_movement(self, dist_grid):

    # TODO: Check first with no double coverage allowed, then check again with DC allowed
    possible_translations = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    possible_rotations = [(Direction.UP, Direction.LEFT), (Direction.UP, Direction.RIGHT), (Direction.DOWN, Direction.LEFT), (Direction.DOWN, Direction.RIGHT)]

    shift_movement_costs = [self.get_shifted_cost(i, dist_grid) for i in possible_translations]
    rotate_movement_costs = [self.get_rotated_cost(i, j, dist_grid) for i,j in possible_rotations]

    movement_costs = shift_movement_costs + rotate_movement_costs

    best_move = movement_costs.index(max(movement_costs))

    print(movement_costs)
    contains_valid_move = any(map(lambda  x: x > 0, movement_costs))

    if not contains_valid_move:
      return -1, -1, False

    if best_move < len(possible_translations):
      next_move = self.cur_pos.shift(possible_translations[best_move])
      return next_move.x, next_move.y, False
    else:
      best_move -= len(possible_translations)

      temp_y = self.size_x
      temp_x = self.size_y

      shifted_pos = self.cur_pos

      for i in range(abs(temp_y // 2 - temp_x // 2)):
        shifted_pos = shifted_pos.shift(possible_rotations[best_move][0])

      for i in range(abs(temp_y // 2 - temp_x // 2)):
        shifted_pos = shifted_pos.shift(possible_rotations[best_move][1])

      return shifted_pos.x, shifted_pos.y, True
    
  def get_shifted_cost(self, shift_dir, dist_grid):
    shifted_pos = self.cur_pos.shift(shift_dir)
    shifted_corners = self.compute_corners(shifted_pos, self.size_x, self.size_y)

    cost = 0
    counted_cells = 0

    if shift_dir == Direction.UP:
      temp_y = shifted_corners[0].y

      for i in range(shifted_corners[0].x, shifted_corners[1].x + 1):

        # If any position of the footprint is not in bounds or has been visited this is not a valid position
        if not dist_grid.inBounds(i, temp_y):
          return -1

        if dist_grid[i, temp_y] != 0:
          cost = max(cost,dist_grid[i, temp_y])
          counted_cells += 1
    
    elif shift_dir == Direction.DOWN:
      temp_y = shifted_corners[2].y

      for i in range(shifted_corners[0].x, shifted_corners[1].x + 1):

        # If any position of the footprint is not in bounds or has been visited this is not a valid position
        if not dist_grid.inBounds(i, temp_y):
          return -1

        if dist_grid[i, temp_y] != 0:
          cost = max(cost,dist_grid[i, temp_y])
          counted_cells += 1

    elif shift_dir == Direction.LEFT:
      temp_x = shifted_corners[0].x

      for j in range(shifted_corners[0].y, shifted_corners[2].y + 1):

        # If any position of the footprint is not in bounds or has been visited this is not a valid position
        if not dist_grid.inBounds(temp_x, j):
          return -1
        
        if dist_grid[temp_x, j] != 0:
          cost = max(cost,dist_grid[temp_x, j])
          counted_cells += 1

    elif shift_dir == Direction.RIGHT:
      temp_x = shifted_corners[1].x

      for j in range(shifted_corners[0].y, shifted_corners[2].y + 1):

        # If any position of the footprint is not in bounds or has been visited this is not a valid position
        if not dist_grid.inBounds(temp_x, j):
          return -1

        if dist_grid[temp_x, j] != 0:
          cost = max(cost,dist_grid[temp_x, j])
          counted_cells += 1

    if counted_cells == 0:
      return -1

    # if (shift_dir == Direction.UP or shift_dir == Direction.DOWN) and self.size_x < self.size_y:
    #   return (cost/counted_cells) / 5
    # elif (shift_dir == Direction.LEFT or shift_dir == Direction.RIGHT) and self.size_y < self.size_x:
    #   return (cost/counted_cells) / 5
    # else:
    return cost
      
  def get_rotated_cost(self, vertical_dir, horizontal_dir, dist_grid):
    temp_y = self.size_x
    temp_x = self.size_y

    assert vertical_dir == Direction.UP or vertical_dir == Direction.DOWN
    assert horizontal_dir == Direction.LEFT or horizontal_dir == Direction.RIGHT

    shifted_pos = self.cur_pos

    for i in range(abs(temp_y // 2 - temp_x // 2)):
      shifted_pos = shifted_pos.shift(vertical_dir)

    for j in range(abs(temp_y // 2 - temp_x // 2)):
      shifted_pos = shifted_pos.shift(horizontal_dir)


    # Compute the cost
    shifted_corners = self.compute_corners(shifted_pos, temp_x, temp_y)

    cost = 0
    counted_cells = 0

    for i in range(shifted_corners[0].x, shifted_corners[1].x + 1):
      for j in range(shifted_corners[0].y, shifted_corners[2].y + 1):

        # If any position of the footprint is not in bounds or has been visited this is not a valid position
        if not dist_grid.inBounds(i, j):
          return -1

        if dist_grid[i, j] != 0:
          cost = max(cost,dist_grid[i, j])
          counted_cells += 1

    if counted_cells == 0:
      return -1

    return cost

  def move_to(self, pos_x, pos_y, rotate = False):
    self.cur_pos = Point(pos_x, pos_y)

    # Swap dimensions if we rotate
    if rotate:
      self.size_x, self.size_y = self.size_y, self.size_x

  def lawnmower(self, area_size_x, area_size_y, overall_direction, initial_direction):
    path = [self.cur_pos]
    sweep_length = 0
    number_of_sweeps = 0
    remaining = 0

    temp_pos = self.cur_pos

    # overall direction and initial direction must be perpindicular
    assert overall_direction != initial_direction
    assert overall_direction.opposite() != initial_direction

    if initial_direction == Direction.DOWN or initial_direction == Direction.UP:
      sweep_length = area_size_y - self.size_y
      number_of_sweeps = area_size_x // self.size_x
      remaining = area_size_x % self.size_x
    else:
      sweep_length = area_size_x - self.size_x
      number_of_sweeps = area_size_y // self.size_y
      remaining = area_size_y % self.size_y

    for count, n in enumerate(range(number_of_sweeps)):

      # Move along sweep line
      for i in range(sweep_length):
        temp_pos = temp_pos.shift(initial_direction)
        path.append(temp_pos)

      initial_direction = initial_direction.opposite()

      # Exit here on last movement (dont shift when we reach end)
      if count == number_of_sweeps - 1:
        break

      # Shift to next sweep line
      for j in range(max(self.size_x, self.size_y)):
        temp_pos = temp_pos.shift(overall_direction)
        path.append(temp_pos)

    # This will cover the case where the rectangle is not a perfect division of the length of the sweeper
    # NOTE: This causes double coverage along the entire sweep line
    if remaining != 0:
      # Shift to next sweep line
      for j in range(remaining):
        temp_pos = temp_pos.shift(overall_direction)
        path.append(temp_pos)
      
      for i in range(sweep_length):
        temp_pos = temp_pos.shift(initial_direction)
        path.append(temp_pos)

    return path

