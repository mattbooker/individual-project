import heapq as h
import math
import random
import time
from collections import defaultdict, deque

import numpy as np
from matplotlib import pyplot as plt

from occupancyGrid import OccupancyGrid
from utils import raytrace


class CCRA:

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

  def nhood4(self, x, y):
    result = []

    if x >= 1 and self.dist_grid[x-1, y] != -1:
      result.append((x - 1, y))

    if x < self.dist_grid.size_x - 1 and self.dist_grid[x+1, y] != -1:
      result.append((x + 1, y))

    if y >= 1 and self.dist_grid[x, y-1] != -1:
      result.append((x, y - 1))

    if y < self.dist_grid.size_y - 1 and self.dist_grid[x, y+1] != -1:
      result.append((x, y + 1))

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

  def augmentedNhood4(self, current_pos, current_layer):
    x = self.block_size_x
    location_offsets_unrotated = [(0, self.block_size_y), 
                                  (0, -self.block_size_y), 
                                  (self.block_size_x,0), 
                                  (-self.block_size_x, 0)]

    angle = current_layer * math.pi/8
    locations = []

    for i, j in location_offsets_unrotated:

      ox, oy = current_pos
      x = current_pos[0] + i
      y = current_pos[1] + j
      rotated_x = int(round((x - ox) * math.cos(angle) - (y - oy) * math.sin(angle) + ox,0))
      rotated_y = int(round((x - ox) * math.sin(angle) + (y - oy) * math.cos(angle) + oy,0))

      # Skip if out of x-bounds
      if rotated_x < 0 or rotated_x >= self.dist_grid.size_x:
        continue

      # Skip if out of y-bounds
      if rotated_y < 0 or rotated_y >= self.dist_grid.size_x:
        continue

      # Only add if it is not located in an object and can be reached from current position
      if self.dist_grid[rotated_x, rotated_y] != -1:
        end_layer = self.endLayer(current_pos, current_layer, (rotated_x, rotated_y))

        if end_layer == None:
          continue
        else:
          locations.append(((rotated_x, rotated_y), end_layer))

    return locations

  def endLayer(self, current_pos, current_layer, goal_pos):
    # Need to check if we can move from current pos to goal pos
    # In order for two adjacent cells to be pathable
    # the cells must have an unoccupied layer 1 shift apart

    # Thus for a path to be valid all adjacent cells must satisfy the above

    # No possible orientations can be reached at goal
    if self.oriented_occ_grid[goal_pos] == 255:
      return None

    # Get the direct path from current to goal
    traversed_cells = raytrace(*current_pos, *goal_pos)

    # Create this structure so we can efficiently get the next cell
    next_cell = dict()

    for num, cell in enumerate(traversed_cells[:-1]):
      next_cell[cell] = traversed_cells[num + 1]


    # depth first search through the path to find a suitable path
    dfs = []
    dfs.append((current_pos, current_layer))
    visited = set()

    while len(dfs) > 0:
      pos, layer = dfs.pop()

      if (pos, layer) in visited:
        continue
      else:
        visited.add((pos, layer))
      
      # Return True if we have reached the goal
      if pos == goal_pos:
        return layer

      # Skip if layer is completely blocked
      if self.oriented_occ_grid[pos] == 255:
        continue

      next_pos = next_cell[pos]
        
      # Check adjacent and current layers of next position to see if it can be reached from here
      for i in [-1, 1, 0]:
        next_layer = (layer + i) % 8
        
        if self.oriented_occ_grid[next_pos] & 2**next_layer == 0:
          # print(pos, layer, next_pos, next_layer)
          dfs.append((next_pos, next_layer))

    # dfs didnt find a suitable path so return false
    return None

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

  def rotatedFilledCells(self, angle, size_x=None, size_y=None):

    if size_x is None and size_y is None:
      half_size_x = self.block_size_x//2
      half_size_y = self.block_size_y//2
    else:
      half_size_x = size_x//2
      half_size_y = size_y//2

    corners = [(0,0)] * 4

    # Top left, top right, bottom right, bottom left (clockwise ordering of points)
    # Corners of the block centered at 0,0
    corners[0] = (-half_size_x, -half_size_y)
    corners[1] = (half_size_x, -half_size_y)
    corners[2] = (half_size_x, half_size_y)
    corners[3] = (-half_size_x, half_size_y)

    for i in range(4):
      x, y = corners[i]
      rotated_x = x * math.cos(angle) - y * math.sin(angle)
      rotated_y = x * math.sin(angle) + y * math.cos(angle)
      # print(rotated_x, rotated_y)
      corners[i] = (int(round(rotated_x, 0)), int(round(rotated_y, 0)))

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

  def moveToUnvisited(self, current_pos, current_layer):
    bfs = deque()
    bfs.append((current_pos, current_layer))
    visited = set()
    prev = dict()
    dist = defaultdict(lambda : float('inf'))

    dist[current_pos] = 0
    prev[current_pos] = None

    while len(bfs) > 0:
      current_pos, current_layer = bfs.popleft()

      if current_pos in visited:
        continue
      else:
        visited.add(current_pos)

      for nbr in self.nhood4(*current_pos):
        next_layer = self.getNextLayer(current_pos, current_layer, nbr)

        # If no movement to nbr exists skip
        if next_layer == None:
          continue

        # If we found a nonzero space then return the path to here
        if self.dist_grid[nbr] != 0:
          path = [(nbr, next_layer), (current_pos, current_layer)]

          pos = current_pos
          layer = current_layer

          while prev[pos] != None:
            next_pos, layer = prev[pos]
            path.append((next_pos, layer))
            pos = next_pos

          return path[::-1]

          

        new_dist = 1 + dist[current_pos]

        if new_dist < dist[nbr]:
          dist[nbr] = new_dist
          prev[nbr] = (current_pos, current_layer)

        bfs.append((nbr,next_layer))
    
    return []

  def getVisitedCells(self, current_pos, current_layer):
    angle = current_layer * math.pi/8
    offset_cells = self.rotatedFilledCells(angle)

    visited = []

    for i, j in offset_cells:
      visited.append((current_pos[0] + i, current_pos[1] + j))
      
    return visited

  def getOccupiedCells(self, current_pos, current_layer):
    occupied_size_x = self.block_size_x + 2 * (self.block_size_x//2)
    occupied_size_y = self.block_size_y + 2 * (self.block_size_y//2)

    angle = current_layer * math.pi/8

    offset_cells = self.rotatedFilledCells(angle, occupied_size_x, occupied_size_y)

    occupied = []

    for i, j in offset_cells:
      occupied.append((current_pos[0] + i, current_pos[1] + j))

    return occupied

  def getPath(self):
    # Compute configuration space
    self.computeCSpace()

    # TODO: Not the best way to do this -> want to start at most explorable point
    # Generate a random start point within the cspace
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

    # occupied = set()
    c = 0

    # Follow the path of min cost
    while True:
      current_cost = self.dist_grid[current_pos]

      min_cost = float('inf')
      best_next_pos = None
      best_next_layer = None

      visited = self.getVisitedCells(current_pos, current_layer)
      # occupied.update(self.getOccupiedCells(current_pos, current_layer))

      # Mark visited cells as visited
      for i, j in visited:
        self.dist_grid[i,j] = 0

      for nbr, layer in self.augmentedNhood4(current_pos, current_layer):
        
        if self.dist_grid[nbr] > 0 and self.dist_grid[nbr] < min_cost:
          min_cost = self.dist_grid[nbr]
          best_next_pos = nbr
          best_next_layer = layer

      path.append((current_pos, current_layer))
      self.dist_grid[current_pos] = 0

      # c += 1
      # if c:
      #   plt.imshow(masked_dist_grid)
      #   plt.pause(0.1)
      #   plt.clf()
      #   c = 0

      # If the surrounding cells are all 0 (i.e. visited) then look to move to next best area
      if best_next_pos == None:
        movement_to_unvisited = self.moveToUnvisited(current_pos, current_layer)

        # If we cant reach, return path
        if len(movement_to_unvisited) == 0:
          return self.process_path(path)
        else:
          path.extend(movement_to_unvisited)
      
        current_pos, current_layer = path[-1]

      else:
        current_pos = best_next_pos
        current_layer = best_next_layer

  def process_path(self, path):
    final_path = []

    prev_pos, prev_layer = path[0]
    for pos, layer in path:

      # Check if we need to interpolate between two points
      if abs(pos[0] - prev_pos[0]) > 1 or abs(pos[1] - prev_pos[1]) > 1:
        layer_at_cell = self.interpolatePath(prev_pos, prev_layer, pos, layer)

        assert layer_at_cell != None, "Could not find an intermediate path"

        cells = raytrace(*prev_pos, *pos)

        for c in cells:
          final_path.append((c, layer_at_cell[c]))
      else:
        final_path.append((pos, layer))

      prev_pos = pos
      prev_layer = layer

    return final_path

  def interpolatePath(self, current_pos, current_layer, goal_pos, goal_layer):
    # Get the direct path from current to goal
    traversed_cells = raytrace(*current_pos, *goal_pos)

    # Create this structure so we can efficiently get the next cell
    next_cell = dict()

    for num, cell in enumerate(traversed_cells[:-1]):
      next_cell[cell] = traversed_cells[num + 1]

    # depth first search through the path to find a suitable path
    dfs = []
    dfs.append((current_pos, current_layer))
    visited = set()
    layer_at_cell = dict()

    while len(dfs) > 0:
      pos, layer = dfs.pop()

      if (pos, layer) in visited:
        continue
      else:
        visited.add((pos, layer))

      layer_at_cell[pos] = layer
      
      # Return True if we have reached the goal
      if pos == goal_pos and layer == goal_layer:
        return layer_at_cell

      next_pos = next_cell[pos]
        
      # Check adjacent and current layers of next position to see if it can be reached from here
      for i in [-1, 1, 0]:
        next_layer = (layer + i) % 8
        
        if self.oriented_occ_grid[next_pos] & 2**next_layer == 0:
          # print(pos, layer, next_pos, next_layer)
          dfs.append((next_pos, next_layer))

    # dfs didnt find a suitable path so return false
    return None
