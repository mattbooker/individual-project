import heapq as h
import math
import time
from collections import defaultdict, deque

import numpy as np
from scipy.spatial import distance_matrix
from intervaltree import IntervalTree

from utils import Direction, Point, Pose, raytrace

from matplotlib import pyplot as plt


class SubmapPlanner:
  def __init__(self, occ_grid, block_size_x, block_size_y):

    self.occ_grid = occ_grid
    self.oriented_occ_grid = occ_grid.clone()
    self.oriented_occ_grid.grid = self.oriented_occ_grid.grid.astype(np.uint8)
    self.oriented_occ_grid.grid[np.nonzero(self.oriented_occ_grid.grid)] = 255

    self.initial_block_size_x = block_size_x
    self.initial_block_size_y = block_size_y
    self.block_size_x = block_size_x
    self.block_size_y = block_size_y

    self.computeCSpace()

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

  def createGraphFromSubmaps(self, submaps):
    x_intervals = IntervalTree()
    y_intervals = IntervalTree()

    for num, submap in enumerate(submaps):
      x_intervals[submap.min_x: submap.max_x + 1] = num
      y_intervals[submap.min_y: submap.max_y + 1] = num

    # print(x_intervals)
    # print(y_intervals)

    adj_list = [[] for i in range(len(submaps))]

    for num, a in enumerate(submaps):
    
      left_edge = x_intervals[a.min_x - 1]
      right_edge = x_intervals[a.max_x + 1]
      top_edge = y_intervals[a.min_y - 1]
      bottom_edge = y_intervals[a.max_y + 1]

      # Find all rectangles connected to the left edge
      for interval in left_edge:
        b = submaps[interval.data]

        if a.min_y >= b.min_y:
          if a.min_y <= b.max_y:
            # Case A or C
            adj_list[num].append(interval.data)
        elif a.max_y >= b.min_y:
          # Case B
          adj_list[num].append(interval.data)
        elif a.max_y >= b.max_y:
          # Case D
          adj_list[num].append(interval.data)

      # Right edge
      for interval in right_edge:
        b = submaps[interval.data]

        if a.min_y >= b.min_y:
          if a.min_y <= b.max_y:
            # Case A or C
            adj_list[num].append(interval.data)
        elif a.max_y >= b.min_y:
          # Case B
          adj_list[num].append(interval.data)
        elif a.max_y >= b.max_y:
          # Case D
          adj_list[num].append(interval.data)

      # Top Edge
      for interval in top_edge:
        b = submaps[interval.data]

        if a.min_x >= b.min_x:
          if a.min_x <= b.max_x:
            # Case A or C
            adj_list[num].append(interval.data)
        elif a.max_x >= b.min_x:
          # Case B
          adj_list[num].append(interval.data)
        elif a.max_x >= b.max_x:
          # Case D
          adj_list[num].append(interval.data)

      # Bottom Edge
      for interval in bottom_edge:
        b = submaps[interval.data]

        if a.min_x >= b.min_x:
          if a.min_x <= b.max_x:
            # Case A or C
            adj_list[num].append(interval.data)
        elif a.max_x >= b.min_x:
          # Case B
          adj_list[num].append(interval.data)
        elif a.max_x >= b.max_x:
          # Case D
          adj_list[num].append(interval.data)

    return adj_list

  def createDistMat(self, adj_list, submaps):
    def bfs(start, end):
      bfs = deque()
      bfs.append(start)

      prev = dict()
      dist = defaultdict(lambda: float('inf'))
      visited = set()

      prev[start] = None
      dist[start] = 0

      while len(bfs) > 0:
        cur = bfs.popleft()

        if cur in visited:
          continue
        else:
          visited.add(cur)

        if cur == end:
          path = []
          tmp = cur

          while prev[tmp] != None:
            path.append(tmp)
            tmp = prev[tmp]

          return [start] + path[::-1]

        for nbr in adj_list[cur]:
          new_dist = dist[cur] + dist_mat[cur][nbr]

          if new_dist < dist[nbr]:
            dist[nbr] = new_dist
            prev[nbr] = cur
            bfs.append(nbr)

      return []

    n = len(adj_list)

    points = []
    for s in submaps:
      points.append([s.centre_x, s.centre_y])

    dist_mat = distance_matrix(points, points)
    updated_dist_mat = dist_mat.copy()

    for i in range(n):
      for j in range(n):
        if i == j:
          continue

        if j not in adj_list[i]:
          shortest_path = bfs(i, j)
          
          new_dist = 0
          for k in range(len(shortest_path) - 1):
            cur = shortest_path[k]
            nxt = shortest_path[k+1]
            new_dist += dist_mat[cur][nxt]

          updated_dist_mat[i][j] = new_dist

    return updated_dist_mat

  def nearestNeighbourTSP(self, adj_list, submaps):
    
    dist_mat = self.createDistMat(adj_list, submaps)

    # Start at submap 0
    current_node = 0

    n = len(adj_list)
    unvisited_nodes = [i for i in range(1, n)]
    visited_nodes = [current_node]

    while n > len(visited_nodes):      
      # Pick the lowest cost unvisited
      cheapest_node = unvisited_nodes[np.argmin(dist_mat[current_node][unvisited_nodes])]

      # Remove the chosen node from unvisited, add to visited, and update to current
      unvisited_nodes.remove(cheapest_node)
      current_node = cheapest_node
      visited_nodes.append(current_node)

    return visited_nodes

  def rotationLayer(self, submap, overall_direction, initial_direction):

    layer = 0

    # if initial_direction == Direction.UP or initial_direction == Direction.DOWN:
    #   layer = 0
    if initial_direction == Direction.LEFT or initial_direction == Direction.RIGHT:
      layer = 4

    # if max(self.block_size_x, self.block_size_y) > min(submap.size_x, submap.size_y):
    #   layer = (layer + 4) % 8

    return layer 

  def rotationFix(self, cur_layer):
    
    if cur_layer == 0:
      self.block_size_x = self.initial_block_size_x
      self.block_size_y = self.initial_block_size_y
    elif cur_layer == 4:
      self.block_size_x = self.initial_block_size_y
      self.block_size_y = self.initial_block_size_x

  def lawnmower(self, start_point, start_layer, submap): 
    
    assert (start_layer == 0 or start_layer == 4)

    path = [(start_point.x, start_point.y, start_layer)]

    overall_direction = submap.overall_direction
    initial_direction = submap.initial_direction
    
    sweep_length = 0
    number_of_sweeps = 0
    remaining = 0

    temp_pos = start_point
    
    block_shift_side = self.block_size_x if overall_direction.isHorizontal() else self.block_size_y
    submap_shift_side = submap.size_x if overall_direction.isHorizontal() else submap.size_y

    block_sweep_side = self.block_size_x if initial_direction.isHorizontal() else self.block_size_y
    submap_sweep_side = submap.size_x if initial_direction.isHorizontal() else submap.size_y

    sweep_length = submap_sweep_side - block_sweep_side
    number_of_sweeps = submap_shift_side // block_shift_side
    remaining = submap_shift_side % block_shift_side

    # TODO: This only occurs when the submap is smaller than our block -> is it needed?
    if number_of_sweeps == 0:
      number_of_sweeps = 1
      remaining = 0

    for count, n in enumerate(range(number_of_sweeps)):

      # Move along sweep line
      for i in range(sweep_length):
        temp_pos = temp_pos.shift(initial_direction)
        path.append((temp_pos.x, temp_pos.y, start_layer))

      initial_direction = initial_direction.opposite()

      # Exit here on last movement (dont shift when we reach end)
      if count == number_of_sweeps - 1:
        break

      # Shift to next sweep line
      for j in range(max(self.block_size_x, self.block_size_y)):
        temp_pos = temp_pos.shift(overall_direction)
        path.append((temp_pos.x, temp_pos.y, start_layer))

    # This will cover the case where the rectangle is not a perfect division of the length of the sweeper
    # NOTE: This causes double coverage along the entire sweep line
    if remaining != 0:
      # Shift to next sweep line
      for j in range(remaining):
        temp_pos = temp_pos.shift(overall_direction)
        path.append((temp_pos.x, temp_pos.y, start_layer))
      
      for i in range(sweep_length):
        temp_pos = temp_pos.shift(initial_direction)
        path.append((temp_pos.x, temp_pos.y, start_layer))

    return path

  def getSweepDirection(self, submap, start_point):

    # Notes:
    # Want maximum dim of block to face min dim of submap

    # min(block) > min(sub) -- This case should never happen -> filter out beforehand by submap processor
    # max(block) > max(sub) -- same as above except there is a possibility of cleaning it given theres some space on one of the sides

    overall_direction = None
    initial_direction = None

    diff_to_top = start_point.y - submap.min_y
    diff_to_bot = submap.max_y - start_point.y

    diff_to_left = start_point.x - submap.min_x
    diff_to_right = submap.max_x - start_point.x

    if submap.size_x >= submap.size_y:
      # Overall direction is UP/DOWN
      # Initial direction is LEFT/RIGHT

      if diff_to_top <= diff_to_bot:
        overall_direction = Direction.DOWN
      else:
        overall_direction = Direction.UP

      if diff_to_left <= diff_to_right:
        initial_direction = Direction.RIGHT
      else:
        initial_direction = Direction.LEFT
    
    else:
      # Overall direction is LEFT/RIGHT
      # Initial direction is UP/DOWN

      if diff_to_top <= diff_to_bot:
        initial_direction = Direction.DOWN
      else:
        initial_direction = Direction.UP

      if diff_to_left <= diff_to_right:
        overall_direction = Direction.RIGHT
      else:
        overall_direction = Direction.LEFT


    # If the block doesnt fit in the submap then we need change how we choose where to go
    if max(self.block_size_x, self.block_size_y) > min(submap.size_x, submap.size_y):
      # Swap the direction orders
      overall_direction, initial_direction = initial_direction, overall_direction

    # overall direction and initial direction must be perpindicular
    assert overall_direction != initial_direction
    assert overall_direction.opposite() != initial_direction

    return (overall_direction, initial_direction)

  def pathToNextSubmap(self, cur_pos, cur_layer, next_submap):
    corner_motions = []
    corner_targets = []
    path = []

    if next_submap.is_rectangle:
      for c in next_submap.corners:
        overall_direction, initial_direction = self.getSweepDirection(next_submap, c)
        corner_motions.append((overall_direction, initial_direction))

        goal = self.refineSubmapStartpoint(c, overall_direction, initial_direction)
        goal_layer = self.rotationLayer(next_submap, overall_direction, initial_direction)

        corner_targets.append((goal, goal_layer))

    else:
      for c in next_submap.corners:
        occupied_layers = self.oriented_occ_grid[c.x, c.y]
        
        if occupied_layers == 255:
          continue

        for layer in range(8):
          if occupied_layers & 2**layer == 0:
            corner_targets.append((c,layer))

    path = self.dijkstra(cur_pos, cur_layer, corner_targets)

    if len(path) != 0 and next_submap.is_rectangle:
      selected_corner = (Point(path[-1][0], path[-1][1]), path[-1][2])
      ind = corner_targets.index(selected_corner)

      # save the directions to the submap
      next_submap.overall_direction = corner_motions[ind][0]
      next_submap.initial_direction = corner_motions[ind][1]

      self.rotationFix(path[-1][2])

    return path

  def dijkstra(self, start_pos, start_layer, targets):

    # Ensure a at least one corner is unoccupied
    none_free = True
    for g_pos, g_layer in targets:
      if self.oriented_occ_grid[g_pos.x, g_pos.y] & 2**g_layer == 0:
        none_free = False
    
    if none_free:
      return []

    visited = [set() for i in range(8)]
    prev = [dict() for i in range(8)]
    distance =  [defaultdict(lambda: float('inf')) for i in range(8)]
    pq = []

    distance[start_layer][start_pos] = 0
    prev[start_layer][start_pos] = None

    h.heappush(pq, (0, (start_pos, start_layer)))
    
    while len(pq) > 0:

      current_dist, (current_pos, current_layer) = h.heappop(pq)

      # Check if one of the goals has been reached
      reached_goal = False
      for g_pos, g_layer in targets:

        if current_pos == g_pos and current_layer == g_layer:
          reached_goal = True
          break

      # If we reached the goal then extract the path and reverse it
      if reached_goal:
        path = [(current_pos.x, current_pos.y, current_layer)]

        pos = current_pos
        layer = current_layer

        while prev[layer][pos] != None:
          pos, layer = prev[layer][pos]
          path.append((pos.x, pos.y, layer))
        
        # Reverse the path since it will be from goal to start and we want start to goal
        return path[::-1]

      # If we havent visited this point then add to visited. If we have visited then no point checking it again
      if current_pos not in visited[current_layer]:
        visited[current_layer].add(current_pos)
      else:
        continue

      prev_layer = (current_layer - 1) % 8
      next_layer = (current_layer + 1) % 8
      candidates = list(zip(current_pos.nhood4(), [current_layer] * 4)) + [(current_pos, prev_layer), (current_pos, next_layer)]

      # Check distances to all neighbouring points
      for nbr_pos, nbr_layer in candidates:

        # Skip if point is not in bounds or is occupied
        if (not self.oriented_occ_grid.inBounds(nbr_pos.x, nbr_pos.y)) or (self.oriented_occ_grid[nbr_pos.x, nbr_pos.y] & 2**nbr_layer != 0):
          continue


        # If not previously visited then compute the current distance and compare to the stored distance for that point
        if nbr_pos not in visited[nbr_layer]:
          new_dist = current_dist + 1

          if new_dist < distance[nbr_layer][nbr_pos]:
            distance[nbr_layer][nbr_pos] = new_dist
            prev[nbr_layer][nbr_pos] = (current_pos, current_layer)


            h.heappush(pq, (new_dist, (nbr_pos, nbr_layer)))

    return []

  def refineSubmapStartpoint(self, target_corner, overall_direction, initial_direction):

    target_point = target_corner.copy()
    
    shift_value_x = round((self.block_size_x - 0.5) / 2)
    shift_value_y = round((self.block_size_y - 0.5) / 2)

    if overall_direction == Direction.UP:
      if initial_direction == Direction.LEFT:
        target_point.x += -min(shift_value_x, shift_value_y)
        target_point.y += -max(shift_value_x, shift_value_y)
      else:
        target_point.x += min(shift_value_x, shift_value_y)
        target_point.y += -max(shift_value_x, shift_value_y)
    elif overall_direction == Direction.DOWN:
      if initial_direction == Direction.LEFT:
        target_point.x += -min(shift_value_x, shift_value_y)
        target_point.y += max(shift_value_x, shift_value_y)
      else:
        target_point.x += min(shift_value_x, shift_value_y)
        target_point.y += max(shift_value_x, shift_value_y)
    elif overall_direction == Direction.LEFT:
      if initial_direction == Direction.UP:
        target_point.x += -max(shift_value_x, shift_value_y)
        target_point.y += -min(shift_value_x, shift_value_y)
      else:
        target_point.x += -max(shift_value_x, shift_value_y)
        target_point.y += min(shift_value_x, shift_value_y)
    else:
      if initial_direction == Direction.UP:
        target_point.x += max(shift_value_x, shift_value_y)
        target_point.y += -min(shift_value_x, shift_value_y)
      else:
        target_point.x += max(shift_value_x, shift_value_y)
        target_point.y += min(shift_value_x, shift_value_y)

    return target_point

  def getVisitedCells(self, cur_pos, cur_layer):
    angle = cur_layer * math.pi/8
    offset_cells = self.rotatedFilledCells(angle)

    visited = []

    for i, j in offset_cells:
      visited.append((cur_pos[0] + i, cur_pos[1] + j))
      
    return visited

  def endLayer(self, cur_pos, cur_layer, goal_pos):
    # Need to check if we can move from current pos to goal pos
    # In order for two adjacent cells to be pathable
    # the cells must have an unoccupied layer 1 shift apart

    # Thus for a path to be valid all adjacent cells must satisfy the above

    # No possible orientations can be reached at goal
    if self.oriented_occ_grid[goal_pos] == 255:
      return None

    # Get the direct path from current to goal
    traversed_cells = raytrace(*cur_pos, *goal_pos)

    # Create this structure so we can efficiently get the next cell
    next_cell = dict()

    for num, cell in enumerate(traversed_cells[:-1]):
      next_cell[cell] = traversed_cells[num + 1]


    # depth first search through the path to find a suitable path
    dfs = []
    dfs.append((cur_pos, cur_layer))
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

  def augmentedNhood4(self, cur_pos, cur_layer, dist_grid):
    x = self.block_size_x
    location_offsets_unrotated = [(0, self.block_size_y), 
                                  (0, -self.block_size_y), 
                                  (self.block_size_x,0), 
                                  (-self.block_size_x, 0)]

    angle = cur_layer * math.pi/8
    locations = []

    for i, j in location_offsets_unrotated:

      ox, oy = cur_pos
      x = cur_pos[0] + i
      y = cur_pos[1] + j
      rotated_x = int(round((x - ox) * math.cos(angle) - (y - oy) * math.sin(angle) + ox,0))
      rotated_y = int(round((x - ox) * math.sin(angle) + (y - oy) * math.cos(angle) + oy,0))

      # Skip if out of x-bounds
      if not dist_grid.inBounds(rotated_x, rotated_y):
        continue

      # Only add if it is not located in an object and can be reached from current position
      if dist_grid[rotated_x, rotated_y] != -1:
        end_layer = self.endLayer(cur_pos, cur_layer, (rotated_x, rotated_y))

        if end_layer == None:
          continue
        else:
          locations.append(((rotated_x, rotated_y), end_layer))

    return locations

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

  def process_path(self, path):
    final_path = []

    prev_pos_x, prev_pos_y, prev_layer = path[0]
    prev_pos = (prev_pos_x, prev_pos_y)
    for pos_x, pos_y, layer in path:
      pos = (pos_x, pos_y)

      # Check if we need to interpolate between two points
      if abs(pos[0] - prev_pos[0]) > 1 or abs(pos[1] - prev_pos[1]) > 1:
        layer_at_cell = self.interpolatePath(prev_pos, prev_layer, pos, layer)

        assert layer_at_cell != None, "Could not find an intermediate path"

        cells = raytrace(*prev_pos, *pos)

        for c in cells:
          final_path.append((*c, layer_at_cell[c]))
      else:
        final_path.append((*pos, layer))

      prev_pos = pos
      prev_layer = layer

    return final_path

  def nonRectangularSubmapPlan(self, cur_pos, cur_layer, submap):

    # Generate dist grid
    dist_grid = self.occ_grid.clone()
    dist_grid.grid[dist_grid.grid == 1] = -1

    valid_cells = set(submap.range())
    visited = set()

    bfs = deque()
    bfs.append(((cur_pos.x, cur_pos.y), 0))
    
    while len(bfs) > 0:
      pos, cost = bfs.popleft()

      if pos not in visited:
        visited.add(pos)
      else:
        continue

      candidates = [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)]

      for nbr in candidates:
        if not dist_grid.inBounds(*nbr) or (nbr not in valid_cells):
          continue

        new_cost = cost + 1

        if dist_grid[nbr] == 0 or dist_grid[nbr] > new_cost:
          dist_grid[nbr] = new_cost
          bfs.append((nbr, new_cost))
    
    pos = (cur_pos.x, cur_pos.y)
    layer = cur_layer
    path = []
    c = 0

    # Follow the path of min cost
    while True:

      min_cost = float('inf')
      best_next_pos = None
      best_next_layer = None

      visited = self.getVisitedCells(pos, layer)

      for i,j in visited:
        dist_grid[i, j] = 0

      for nbr_pos, nbr_layer in self.augmentedNhood4(pos, layer, dist_grid):
        
        if dist_grid[nbr_pos] > 0 and dist_grid[nbr_pos] < min_cost:
          min_cost = dist_grid[nbr_pos]
          best_next_pos = nbr_pos
          best_next_layer = nbr_layer

      path.append((*pos, layer))
      dist_grid[pos] = 0

      # DEBUGGING VISUALIZATION
      # c += 1
      # if c > 0:
      #   plt.imshow(dist_grid.grid)
      #   plt.pause(0.001)
      #   plt.clf()
      #   c = 0

      # If the surrounding cells are all 0 (i.e. visited) then look to move to next best area
      if best_next_pos == None:
        return self.process_path(path)
      else:
        pos = best_next_pos
        layer = best_next_layer

  def getPath(self, submaps):

    adj_list = self.createGraphFromSubmaps(submaps)
    submap_visit_order = self.nearestNeighbourTSP(adj_list, submaps)

    path = []

    # Setup
    initial_submap = submaps[submap_visit_order[0]]
    cur_pos = initial_submap.corners[0]
    overall_direction, initial_direction = self.getSweepDirection(initial_submap, cur_pos)
    cur_layer = self.rotationLayer(initial_submap, overall_direction, initial_direction)
    cur_pos = self.refineSubmapStartpoint(cur_pos, overall_direction, initial_direction)

    self.rotationFix(cur_layer)

    for i in submap_visit_order:
        
      movement = self.pathToNextSubmap(cur_pos, cur_layer, submaps[i])

      # Occurs when we cant find a path into the submap
      if len(movement) == 0:
        print(f'Skipped submap {i}')
        continue

      path.extend(movement)
      cur_pos = Point(path[-1][0], path[-1][1])
      cur_layer = path[-1][2]

      if submaps[i].is_rectangle:
        path.extend(self.lawnmower(cur_pos, cur_layer, submaps[i]))
      else:
        path.extend(self.nonRectangularSubmapPlan(cur_pos, cur_layer, submaps[i]))

      cur_pos = Point(path[-1][0], path[-1][1])
      cur_layer = path[-1][2]

    return path
