import occupancyGrid
import heapq as h
import numpy as np
from collections import defaultdict

class basicCC:

  def __init__(self, occ_grid, start_pose, target_pose):
    self.occ_grid = occ_grid
    self.dist_grid = np.full((100,100), -1)
    self.start_mx, self.start_my = occ_grid.world_to_map(start_pose[0], start_pose[1])
    self.goal_mx, self.goal_my = occ_grid.world_to_map(target_pose[0], target_pose[1])

  def nhood4(self, x, y):
    result = []

    if x >= 1 and self.occ_grid.get_occupancy(x-1, y) == 0:
      result.append((x - 1, y))

    if x < self.occ_grid.size_x - 1 and self.occ_grid.get_occupancy(x+1, y) == 0:
      result.append((x + 1, y))

    if y >= 1 and self.occ_grid.get_occupancy(x, y-1) == 0:
      result.append((x, y - 1))

    if y < self.occ_grid.size_y - 1 and self.occ_grid.get_occupancy(x, y+1) == 0:
      result.append((x, y + 1))

    return result

  def solve(self):
    self.computeDistGrid()
    return self.get_path()

  def get_path(self):
    visited = set()

    result = np.where(self.dist_grid ==  np.amax(self.dist_grid))
    start_point = (np.asscalar(result[0]), np.asscalar(result[1]))

    result = [start_point]

    current_mx, current_my = start_point

    visited.add((current_mx, current_my))

    while True:

      next_cell = None

      # Get the neighbouring cell with the highest cost that hasnt been visited
      for nx, ny in self.nhood4(current_mx, current_my):
        max_cost = -1

        if (nx, ny) not in visited:
          if self.dist_grid[nx, ny] > max_cost:
            max_cost = self.dist_grid[nx, ny]
            next_cell = (nx, ny)

      if next_cell == None:
        return result
      else:
        result.append(self.occ_grid.map_to_world(next_cell[0], next_cell[1]))

      visited.add(next_cell)
      current_mx, current_my = next_cell


    return result[::-1]

  def computeDistGrid(self):
    prev = dict()
    pq = []

    prev[(self.goal_mx, self.goal_my)] = None
    self.dist_grid[self.goal_mx, self.goal_my] = 0
    h.heappush(pq, (self.goal_mx, self.goal_my))

    while len(pq) != 0:
      current_mx, current_my = h.heappop(pq)
      current_cost = self.dist_grid[current_mx, current_my]

      # Otherwise insert all neighbouring points, unless visited
      for nx, ny in self.nhood4(current_mx, current_my):

        alternate_cost = current_cost + 1

        # If the dist_grid is not popualted at nx,ny or the alternate path is cheaper
        # then we take the current found path instead
        if self.dist_grid[nx, ny] == -1 or alternate_cost < self.dist_grid[nx, ny]:
          self.dist_grid[nx, ny] = alternate_cost
          prev[(nx, ny)] = (current_mx, current_my)

          h.heappush(pq, (nx, ny))