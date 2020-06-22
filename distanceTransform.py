import heapq as h
import math
import random
import time
from collections import defaultdict, deque

import numpy as np
from matplotlib import pyplot as plt

from occupancyGrid import OccupancyGrid
from utils import raytrace


class DistanceTransform:

  def __init__(self, occ_grid, block_size_x, block_size_y):
    # Change obstacles to be -1
    self.dist_grid = occ_grid
    self.dist_grid.grid *= -1

    self.oriented_occ_grid = occ_grid.clone()
    self.oriented_occ_grid.grid = self.oriented_occ_grid.grid.astype(np.uint8)
    self.oriented_occ_grid.grid[np.nonzero(self.oriented_occ_grid.grid)] = 255

    self.block_size_x = block_size_x
    self.block_size_y = block_size_y
    
  def generateRandomStart(self):
    while True:

      result = (random.randint(0, self.dist_grid.size_x - 1),random.randint(0, self.dist_grid.size_y - 1))

      if self.oriented_occ_grid[result] != 255:
        return result

      # x_begin = -self.block_size_x//2
      # x_end = self.block_size_x//2 + 1 if self.block_size_x % 2 == 1 else self.block_size_x//2

      # y_begin = -self.block_size_y//2
      # y_end = self.block_size_y//2 + 1 if self.block_size_y % 2 == 1 else self.block_size_y//2

      # for i in range(x_begin, x_end):
      #   free_space = True

      #   for j in range(y_begin, y_end):
      #     if self.dist_grid.inBounds(result[0] + i,  result[1] + j) and self.dist_grid[result[0] + i, result[1] + j] != 0:
      #       free_space = False
      #       break

      #   if not free_space:
      #     break

      # if free_space:
      #   return result

  def layer_nhood4(self, x, y, current_layer):
    result = []

    if x >= 1 and self.dist_grid[x-1, y] != -1:
      result.append((x - 1, y, current_layer))

    if x < self.dist_grid.size_x - 1 and self.dist_grid[x+1, y] != -1:
      result.append((x + 1, y, current_layer))

    if y >= 1 and self.dist_grid[x, y-1] != -1:
      result.append((x, y - 1, current_layer))

    if y < self.dist_grid.size_y - 1 and self.dist_grid[x, y+1] != -1:
      result.append((x, y + 1, current_layer))

    next_layer = (current_layer + 1) % 8
    prev_layer = (current_layer - 1) % 8
    
    result.append((x, y, next_layer))
    result.append((x, y, prev_layer))

    return result

  def nhood8(self, x, y):
    result = []

    if x >= 1 and self.dist_grid[x-1, y] != -1:
      result.append((x - 1, y))

    if x < self.dist_grid.size_x - 1 and self.dist_grid[x+1, y] != -1:
      result.append((x + 1, y))

    if y >= 1 and self.dist_grid[x, y-1] != -1:
      result.append((x, y - 1))

    if y < self.dist_grid.size_y - 1 and self.dist_grid[x, y+1] != -1:
      result.append((x, y + 1))

    if x >= 1 and y >= 1 and self.dist_grid[x-1, y-1] != -1:
      result.append((x - 1, y - 1))

    if x >= 1 and y < self.dist_grid.size_y - 1 and self.dist_grid[x-1, y+1] != -1:
      result.append((x - 1, y + 1))

    if x < self.dist_grid.size_x - 1 and y >= 1 and self.dist_grid[x+1, y-1] != -1:
      result.append((x + 1, y - 1))

    if x < self.dist_grid.size_x - 1 and y < self.dist_grid.size_y - 1 and self.dist_grid[x+1, y+1] != -1:
      result.append((x + 1, y + 1))

    return result

  def computeDistances(self, start):
    # Use dijkstra
    visited = set()
    prev = dict()
    dist = defaultdict(lambda : float('inf'))
    pq = []

    prev[start] = None
    dist[start] = 0

    # Need to put (cost, pos) into priority queue so that pq is ordered by cost
    h.heappush(pq, (0, start))

    while len(pq) != 0:

      current_cost, current_pos = h.heappop(pq)

      # Check we havent visited this point before
      if current_pos not in visited:
        visited.add(current_pos)
      else:
        continue

      for nbr in self.nhood8(*current_pos):
        new_cost = current_cost + 1

        if new_cost < dist[nbr]:
          dist[nbr] = new_cost
          prev[nbr] = current_pos
          self.dist_grid[nbr] = new_cost

        h.heappush(pq, (dist[nbr], nbr))

  def rotatedFilledCells(self, angle):
    half_size_x = self.block_size_x//2
    half_size_y = self.block_size_y//2

    corners = [(0,0)] * 4

    # Top left, top right, bottom right, bottom left (clockwise ordering of points)
    # Corners of the block centered at 0,0
    corners[0] = (-half_size_x, -half_size_y)
    corners[1] = (half_size_x, -half_size_y)
    corners[2] = (half_size_x, half_size_y)
    corners[3] = (-half_size_x, half_size_y)

    for i in range(4):
      x, y = corners[i]
      rotated_x = x * math.cos(angle) + y * math.sin(angle)
      rotated_y = -x * math.sin(angle) + y * math.cos(angle)
      corners[i] = (int(round(rotated_x, 0)), int(round(rotated_y, 0)))

    # Add bounding polygon cells to a set
    cells = set()
    cells.update(raytrace(*corners[0], *corners[1]))
    cells.update(raytrace(*corners[3], *corners[2]))
    cells.update(raytrace(*corners[0], *corners[3]))
    cells.update(raytrace(*corners[1], *corners[2]))

    # breadth first search from 0,0 the remaining cells
    bfs = deque([(0,0)])
    while len(bfs) > 0:
      current_cell = bfs.popleft()

      # If previously visited then skip
      if current_cell in cells:
        continue
      else:
        cells.add(current_cell)

      # Search the neighbours
      for nbr in [(current_cell[0] + 1, current_cell[1]), (current_cell[0] - 1, current_cell[1]), (current_cell[0], current_cell[1] + 1), (current_cell[0], current_cell[1] - 1)]:
        if nbr not in cells:
          bfs.append(nbr)

    # DEBUGGING
    # longest_edge = max(self.block_size_x, block_size_y)
    # test = np.zeros((30, 30))
    # for i,j in cells:
    #   x = i + 15
    #   y = j + 15
    #   print(x,y)
    #   test[x, y] = 1
      
    # plt.imshow(test)
    # plt.show()
    
    return list(cells)

  def isObstacleFrontier(self, x, y):

    # Obstacle frontier occurs on occupied spaces
    if self.oriented_occ_grid[x, y] == 0:
      return False

    if x >= 1 and self.oriented_occ_grid[x - 1, y] == 0:
      return True

    if x < self.oriented_occ_grid.size_x - 1 and self.oriented_occ_grid[x + 1, y] == 0:
      return True

    if y >= 1 and self.oriented_occ_grid[x, y - 1] == 0:
      return True

    if y < self.oriented_occ_grid.size_y - 1 and self.oriented_occ_grid[x, y + 1] == 0:
      return True

    return False

  def computeCSpace(self):
    # Get obstacle frontiers (+ edge frontiers)
    frontier_cells = []

    # Add obstacle frontier cells
    for i in range(self.oriented_occ_grid.size_x):
      for j in range(self.oriented_occ_grid.size_y):
        if self.isObstacleFrontier(i,j):
          frontier_cells.append((i, j))

    # Add 1 outside edges as frontier cells for x
    for i in range(self.oriented_occ_grid.size_x):
      frontier_cells.append((i, -1))
      frontier_cells.append((i, 100))

    # Add 1 outside edges as frontier cells for y
    for j in range(self.oriented_occ_grid.size_y):
      frontier_cells.append((-1, j))
      frontier_cells.append((100, j))

    combined_offsets = dict()

    # The most expensive part of inflation is iterating the frontier cells, thus
    # to save time we precompute all the rotation offsets and their cumulative values
    # so that we only need to iterate the frontier cells once
    for layer in range(8):
      single_layer_offsets = self.rotatedFilledCells(layer * math.pi/8)

      for i in single_layer_offsets:
        if i not in combined_offsets:
          combined_offsets[i] = 2**layer
        else:
          combined_offsets[i] |= 2**layer

    # Perform inflation on all frontier cells for all rotation angles
    for x, y in frontier_cells:
      for (inflate_x, inflate_y), value in combined_offsets.items():

        if not self.oriented_occ_grid.inBounds(x + inflate_x, y + inflate_y):
          continue

        # NOTE: To save space and time we use 1 grid for all layers, 
        # each cell contains an 8 bit binary, where the 1 is set if
        # that layer is occupied, e.g. if layers 1, 2 and 5 are occupied
        # then the cell will have 00010011 = 19 in it
        self.oriented_occ_grid[x + inflate_x, y + inflate_y] |= value

  def getNextLayer(self, current_pos, current_layer, next_pos):
    # Returns none if cell not reachable otherwise returns layer in next cell
    # Rules for a cell being reachable
    # 1. 1 away from current cell
    # 2. 1 rotation away from any possible configuration in current
    
    occupied_next_layers = self.oriented_occ_grid[next_pos]

    # If everything in the next position is occupied then not reachable
    if occupied_next_layers == (2**8 - 1):
      return None

    next_layer = (current_layer + 1) % 8
    temp_layer = current_layer

    # Check that we can rotate to specific layer and then either rotate into next cell or translate into next cell
    while self.oriented_occ_grid[current_pos] & 2**temp_layer == 0:
      if self.oriented_occ_grid[next_pos] & 2**temp_layer == 0:
        return temp_layer

      if self.oriented_occ_grid[next_pos] & 2**next_layer == 0:
        return next_layer


      next_layer = (next_layer + 1) % 8
      temp_layer = (temp_layer + 1) % 8

      # Stop once we have reached the original starting layer
      if temp_layer == 2**current_layer:
        break

    next_layer = (current_layer - 1) % 8
    temp_layer = current_layer

    # Checks same as above but in opposite direction
    while self.oriented_occ_grid[current_pos] & 2**temp_layer == 0:
      if self.oriented_occ_grid[next_pos] & 2**temp_layer == 0:
        return temp_layer

      if self.oriented_occ_grid[next_pos] & 2**next_layer == 0:
        return next_layer


      next_layer = (next_layer - 1) % 8
      temp_layer = (temp_layer - 1) % 8

      if temp_layer == current_layer:
        break

    return None

  def moveToUnvisited(self, start_pos, start_layer):
    bfs = deque()
    bfs.append((start_pos, start_layer))

    visited = [set() for i in range(8)]
    prev = [dict() for i in range(8)]
    dist =  [defaultdict(lambda: float('inf')) for i in range(8)]

    dist[start_layer][start_pos] = 0
    prev[start_layer][start_pos] = None

    count = 0

    while len(bfs) > 0:
      count += 1
      current_pos, current_layer = bfs.popleft()

      if current_pos in visited[current_layer]:
        continue
      else:
        visited[current_layer].add(current_pos)

      for nbr_x, nbr_y, nbr_layer in self.layer_nhood4(*current_pos, current_layer):
        nbr = (nbr_x, nbr_y)

        # Skip if nbr not free
        if self.oriented_occ_grid[nbr] & 2**nbr_layer != 0:
          continue

        # If we found a nonzero space then return the path to here
        if self.dist_grid[nbr] != 0:
          path = [(nbr, nbr_layer), (current_pos, current_layer)]

          pos = current_pos
          layer = current_layer

          while prev[layer][pos] != None:
            next_pos, next_layer = prev[layer][pos]
            path.append((next_pos, next_layer))
            pos = next_pos
            layer = next_layer

          return path[::-1]

        new_dist = 1 + dist[current_layer][current_pos]

        if new_dist < dist[nbr_layer][nbr]:
          dist[nbr_layer][nbr] = new_dist
          prev[nbr_layer][nbr] = (current_pos, current_layer)

        bfs.append((nbr,nbr_layer))

    return []
        
  def getPath(self, start=None):
    # Compute configuration space
    self.computeCSpace()

    # Generate a random start point within the cspace if no startpoint given
    if start is None:
      start = self.generateRandomStart()

    # Compute distance grid
    self.computeDistances(start)
    
    traversable_mask = self.oriented_occ_grid.grid == 255
    masked_dist_grid =  np.ma.MaskedArray(self.dist_grid.grid, mask=traversable_mask)

    current_pos = start

    # Find the first free layer
    current_layer = 0
    occupied_layers = self.oriented_occ_grid[current_pos]

    while occupied_layers & 1 == 1:
      current_layer += 1
      occupied_layers = occupied_layers >> 1

    path = []
    
    # Follow the path of min cost
    while True:

      min_cost = float('inf')
      best_next_pos = None
      best_next_layer = None

      for nbr_x, nbr_y, nbr_layer in self.layer_nhood4(*current_pos, current_layer):
        nbr = (nbr_x, nbr_y)

        if self.oriented_occ_grid[nbr] & 2**nbr_layer != 0:
          continue
        
        if self.dist_grid[nbr] > 0 and self.dist_grid[nbr] < min_cost:
          min_cost = self.dist_grid[nbr]
          best_next_pos = nbr
          best_next_layer = nbr_layer

      path.append((current_pos, current_layer))
      self.dist_grid[current_pos] = 0

      # If the surrounding cells are all 0 (i.e. visited) then look to move to next best area
      if best_next_pos == None:
        movement_to_unvisited = self.moveToUnvisited(current_pos, current_layer)

        # If we cant reach, return path
        if len(movement_to_unvisited) == 0:
          return path
        else:
          path.extend(movement_to_unvisited)
      
        current_pos, current_layer = path[-1]

      else:
        current_pos = best_next_pos
        current_layer = best_next_layer
