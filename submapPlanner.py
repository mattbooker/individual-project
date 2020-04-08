import heapq as h
import math
from collections import defaultdict

import numpy as np
from intervaltree import IntervalTree

from utils import Direction, Point, Pose


class SubmapPlanner:
  def __init__(self, occ_grid, block_size_x, block_size_y):

    self.occ_grid = occ_grid
    self.inflated_occ_grid = [self.generateInflatedOccGrid(occ_grid, max(block_size_x, block_size_y), min(block_size_x, block_size_y)),
                              self.generateInflatedOccGrid(occ_grid, min(block_size_x, block_size_y), max(block_size_x, block_size_y))]

    self.block_size_x = block_size_x
    self.block_size_y = block_size_y

    self.cur_rot = 0

    self.adj_matrix = None
    self.dp_table = None

    self.leaf_chains = []
    self.idx_map = dict()  

  def generateInflatedOccGrid(self, occ_grid, footprint_x, footprint_y):

    inflated_grid = occ_grid.clone()

    inflate_vertical_edges = False

    if footprint_x == max(footprint_x, footprint_y):
      inflate_vertical_edges = True

    # Iterate over entire costmap
    for i in range(inflated_grid.size_x):
      for j in range(inflated_grid.size_y):

        # If location occupied
        if occ_grid[i, j] == 1:

          # Iterate over footprint shape
          for a in range(-int(footprint_x/2), int(footprint_x/2) + 1):
            for b in range(-int(footprint_y/2), int(footprint_y/2) + 1):

              # If in bounds then set to 1
              if inflated_grid.inBounds(i+a, j+b):
                inflated_grid[i+a, j+b] = 1

    # Fill in sides as well
    for i in range(inflated_grid.size_x):
      for b in range(-int(footprint_y/2), int(footprint_y/2) + 1):
  
        # If in bounds then set to 1
        if inflated_grid.inBounds(i, -1+b):
          inflated_grid[i, -1+b] = 1

        if inflated_grid.inBounds(i, inflated_grid.size_y+b):
          inflated_grid[i, inflated_grid.size_y+b] = 1

    for j in range(inflated_grid.size_y):
      for a in range(-int(footprint_x/2), int(footprint_x/2) + 1):
  
        # If in bounds then set to 1
        if inflated_grid.inBounds(-1+a, j):
          inflated_grid[-1+a, j] = 1

        if inflated_grid.inBounds(inflated_grid.size_x+a, j):
          inflated_grid[inflated_grid.size_x+a, j] = 1

    return inflated_grid

  def createGraphFromSubmaps(self, submaps):
    x_intervals = IntervalTree()
    y_intervals = IntervalTree()

    for num, submap in enumerate(submaps):
      x_intervals[submap.min_x: submap.max_x + 1] = num
      y_intervals[submap.min_y: submap.max_y + 1] = num

    # print(x_intervals)
    # print(y_intervals)

    self.adj_matrix = np.zeros((len(submaps), len(submaps)))

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
            self.adj_matrix[num][interval.data] = 1
        elif a.max_y >= b.min_y:
          # Case B
          self.adj_matrix[num][interval.data] = 1
        elif a.max_y >= b.max_y:
          # Case D
          self.adj_matrix[num][interval.data] = 1

      # Right edge
      for interval in right_edge:
        b = submaps[interval.data]

        if a.min_y >= b.min_y:
          if a.min_y <= b.max_y:
            # Case A or C
            self.adj_matrix[num][interval.data] = 1
        elif a.max_y >= b.min_y:
          # Case B
          self.adj_matrix[num][interval.data] = 1
        elif a.max_y >= b.max_y:
          # Case D
          self.adj_matrix[num][interval.data] = 1

      # Top Edge
      for interval in top_edge:
        b = submaps[interval.data]

        if a.min_x >= b.min_x:
          if a.min_x <= b.max_x:
            # Case A or C
            self.adj_matrix[num][interval.data] = 1
        elif a.max_x >= b.min_x:
          # Case B
          self.adj_matrix[num][interval.data] = 1
        elif a.max_x >= b.max_x:
          # Case D
          self.adj_matrix[num][interval.data] = 1

      # Bottom Edge
      for interval in bottom_edge:
        b = submaps[interval.data]

        if a.min_x >= b.min_x:
          if a.min_x <= b.max_x:
            # Case A or C
            self.adj_matrix[num][interval.data] = 1
        elif a.max_x >= b.min_x:
          # Case B
          self.adj_matrix[num][interval.data] = 1
        elif a.max_x >= b.max_x:
          # Case D
          self.adj_matrix[num][interval.data] = 1

  def findLeafChains(self):
    altered_adj_matrix = np.copy(self.adj_matrix)

    leaf_chain_idx = -1

    for i in range(len(altered_adj_matrix)):
      to_remove = i
      
      first = True

      while to_remove != None:
        if np.count_nonzero(altered_adj_matrix[to_remove]) == 1:
          next_to_remove = np.nonzero(altered_adj_matrix[to_remove])[0]

          altered_adj_matrix[to_remove] = np.zeros(len(self.adj_matrix))
          altered_adj_matrix[:,to_remove] = 0

          if first:
            leaf_chain_idx += 1
            self.leaf_chains.append([])
            first = False

          self.leaf_chains[leaf_chain_idx].append((to_remove, next_to_remove[0]))

          to_remove = next_to_remove[0]
        else:
          to_remove = None

  def seperateLeafChains(self):
    self.findLeafChains()

    # Flattens the leaf chains (Can save time by sorting here if needed)
    removed = [item[0] for sublist in self.leaf_chains for item in sublist]

    sorted_removed = list(sorted(removed, reverse=True))

    new_matrix_size = len(self.adj_matrix) - len(removed)

    # If the new matrix is to have nothing in it return None
    if new_matrix_size == 0:
      return None

    # delete the rows and columns of leaf chains within the adj_matrix
    new_matrix = np.delete(self.adj_matrix, sorted_removed, 0)
    new_matrix = np.delete(new_matrix, sorted_removed, 1)

    # Create the idx_map that helps to retrieve the original indexs of the submaps
    # i.e. self.idx_map[idx in reduced matrix] = idx in original matrix
    non_removed = [i for i in range(len(self.adj_matrix)) if i not in sorted_removed]
    self.idx_map = dict([(i, j) for i,j in zip(range(new_matrix_size), non_removed)])

    return new_matrix

  def heldKarp(self, reduced_adj_matrix):
    num_of_vertices = len(reduced_adj_matrix)
    self.dp_table = np.zeros((num_of_vertices, 2**num_of_vertices))

    for i in range(num_of_vertices):
      self.dp_table[i][2**i] = True
      
    for i in range(2**num_of_vertices):
      for j in range(num_of_vertices):
        if i & (2**j):
          for k in range(num_of_vertices):
            if i & (2**k) and reduced_adj_matrix[k][j] and k != j and self.dp_table[k][i ^ (2**j)]:
              self.dp_table[j][i] = True

    for i in range(num_of_vertices):
      if self.dp_table[i][2**num_of_vertices - 1]:
        return True

    return False

  def getHamiltonianPath(self, reduced_adj_matrix):
    
    start_vertex = 0

    # Get the first vertex that satisfies the hamiltonian path
    for i in range(self.dp_table.shape[0]):
      if self.dp_table[i][-1]:
        start_vertex = i
        break

    current_vertex = start_vertex
    table_bits = 2 ** len(reduced_adj_matrix) - 1

    visited = set()

    path = []

    for i in range(len(reduced_adj_matrix)):

      visited.add(current_vertex)
      path.append(current_vertex)

      neighbours = [idx for idx in range(len(reduced_adj_matrix)) if reduced_adj_matrix[current_vertex][idx] == 1]

      for nbr in neighbours:
        if nbr not in visited:
          if self.dp_table[nbr][table_bits^ 2**current_vertex]:
            table_bits = table_bits ^ 2**current_vertex
            current_vertex = nbr
            break

    # TODO: Remove? used to ensure we get a path that traverses all nodes in the graph
    assert len(path) == len(reduced_adj_matrix)

    return path

  def rotateBlock(self):
    self.block_size_x, self.block_size_y = self.block_size_y, self.block_size_x
    self.cur_rot = int(not self.cur_rot)

  def lawnmower(self, start_point, submap): 
    
    path = [(start_point.x, start_point.y, self.cur_rot)]

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
        path.append((temp_pos.x, temp_pos.y, self.cur_rot))

      initial_direction = initial_direction.opposite()

      # Exit here on last movement (dont shift when we reach end)
      if count == number_of_sweeps - 1:
        break

      # Shift to next sweep line
      for j in range(max(self.block_size_x, self.block_size_y)):
        temp_pos = temp_pos.shift(overall_direction)
        path.append((temp_pos.x, temp_pos.y, self.cur_rot))

    # This will cover the case where the rectangle is not a perfect division of the length of the sweeper
    # NOTE: This causes double coverage along the entire sweep line
    if remaining != 0:
      # Shift to next sweep line
      for j in range(remaining):
        temp_pos = temp_pos.shift(overall_direction)
        path.append((temp_pos.x, temp_pos.y, self.cur_rot))
      
      for i in range(sweep_length):
        temp_pos = temp_pos.shift(initial_direction)
        path.append((temp_pos.x, temp_pos.y, self.cur_rot))

    return path

  def getSweepDirection(self, submap, start_point):

    # TODO: Edge case occurs when we rotate the block but the block is too wide to do so
    # TODO: Edge cases occur when submap is too small to fit it block in specific direction


    # Notes:
    # Want maximum dim of block to face min dim of submap

    # min(block) > min(sub) -- This case should never happen -> filter out beforehand by submap processor
    # max(block) > max(sub) -- same as above except there is a possibility of cleaning it given theres some space on one of the sides

    rotation_required = False
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

      # Check if we need to rotate so largest edge is facing initial direction
      if self.block_size_x > self.block_size_y:
        rotation_required = True
    
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

      # Check if we need to rotate so largest edge is facing initial direction
      if self.block_size_x < self.block_size_y:
        rotation_required = True


    # If the block doesnt fit in the submap then we need change how we choose where to go
    if max(self.block_size_x, self.block_size_y) > min(submap.size_x, submap.size_y):
      # Swap the direction orders
      overall_direction, initial_direction = initial_direction, overall_direction
      rotation_required = not rotation_required

    # overall direction and initial direction must be perpindicular
    assert submap.overall_direction != initial_direction
    assert overall_direction.opposite() != initial_direction

    return (overall_direction, initial_direction, rotation_required)

  def pathToNextSubmap(self, cur_pos, next_submap):

    # Pick the corner of the next submap that is closest to our current position
    target_corner = next_submap.corners[0]
    min_distance = cur_pos.distanceTo(target_corner)

    for i in next_submap.corners[1:]:
      cur_dist = cur_pos.distanceTo(i)

      if cur_dist < min_distance:
        min_distance = cur_dist
        target_corner = i

    current_layer = self.cur_rot

    overall_direction, initial_direction, rotation_required = self.getSweepDirection(next_submap, target_corner)
    goal_point = self.refineSubmapStartpoint(target_corner, overall_direction, initial_direction)

    goal_layer = current_layer

    if rotation_required:
      goal_layer = int(not goal_layer)

    # save the directions to the submap
    next_submap.overall_direction = overall_direction
    next_submap.initial_direction = initial_direction

    print(cur_pos, current_layer, goal_point, goal_layer)

    # If already at goal return goal, otherwise use dijkstra to find path
    if cur_pos == goal_point and current_layer == goal_layer:
      return [(cur_pos.x, cur_pos.y, self.cur_rot)]
    else:
      return self.dijkstra(cur_pos, current_layer, goal_point, goal_layer)

  def dijkstra(self, start_pos, start_layer, goal_pos, goal_layer):

    # assert self.inflated_occ_grid[start_layer][start_pos.x, start_pos.y] == 0
    assert self.inflated_occ_grid[goal_layer][goal_pos.x, goal_pos.y] == 0

    visited = [set(), set()]
    prev = [dict(), dict()]
    distance =  [defaultdict(lambda: float('inf')), defaultdict(lambda : float('inf'))]
    pq = []

    distance[start_layer][start_pos] = 0
    prev[start_layer][start_pos] = None

    h.heappush(pq, (0, (start_pos, start_layer)))
    
    while len(pq) != 0:

      current_dist, (current_pos, current_layer) = h.heappop(pq)

      next_layer = int(not current_layer)

      # If we reached the goal then extract the path and reverse it
      if current_pos == goal_pos and current_layer == goal_layer:
        path = [(goal_pos.x, goal_pos.y, goal_layer)]

        cur_pos, cur_layer = prev[goal_layer][goal_pos]

        while prev[cur_layer][cur_pos] != None:
          path.append((cur_pos.x, cur_pos.y, cur_layer))
          cur_pos, cur_layer = prev[cur_layer][cur_pos]

        # Rotate the block if the start and goal are different orientations
        if goal_layer != start_layer:
          self.rotateBlock()
        
        # Reverse the path since it will be from goal to start and we want start to goal
        return path[::-1]


      # If we havent visited this point then add to visited. If we have visited then no point checking it again
      if current_pos not in visited[current_layer]:
        visited[current_layer].add(current_pos)
      else:
        continue      
    

      # Check distances to all neighbouring points
      for p in current_pos.nhood4():
        
        # Ensure point is in bounds and not occupied
        if (not self.inflated_occ_grid[current_layer].inBounds(p.x, p.y)) or (self.inflated_occ_grid[current_layer][p.x, p.y] != 0):
          continue

        # If not previously visited then compute the current distance and compare to the stored distance for that point
        if p not in visited[current_layer]:
          alternate_dist = current_dist + 1

          if alternate_dist < distance[current_layer][p]:
            distance[current_layer][p] = alternate_dist
            prev[current_layer][p] = (current_pos, current_layer)

            h.heappush(pq, (alternate_dist, (p, current_layer)))

      # Check distance to next layer (this corresponds to making a rotation)
      if self.inflated_occ_grid[next_layer].inBounds(p.x, p.y) and self.inflated_occ_grid[next_layer][p.x, p.y] == 0:
        if p not in visited[next_layer]:
          alternate_dist = distance[current_layer][current_pos] + 50 # Higher cost for rotation

          if alternate_dist < distance[next_layer][p]:
            distance[next_layer][p] = alternate_dist
            prev[next_layer][p] = (current_pos, current_layer)

            h.heappush(pq, (alternate_dist, (p, next_layer)))

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

  def process(self, submaps):

    # TODO: Handle submaps that are too small
    # TODO: Add back in leaf chains? -> or change decomp all together


    self.createGraphFromSubmaps(submaps)

    # TODO: The optimal start and end vertex would be those that are attached to the two longest chains (so we dont need to back track them
    reduced_adj_matrix = self.seperateLeafChains()
    ordered_submaps = []

    if reduced_adj_matrix is not None:
      hamiltonian_path_exists = self.heldKarp(reduced_adj_matrix)
    
      if hamiltonian_path_exists:
        ordered_submaps = self.getHamiltonianPath(reduced_adj_matrix)

      ordered_submaps = list(map(lambda n: self.idx_map[n], ordered_submaps))

    else:
      ordered_submaps = submaps


    path = []

    print(ordered_submaps)
    print(self.leaf_chains)
    print()

    # Setup
    initial_submap = submaps[ordered_submaps[0]]
    # initial_submap = submaps[6]
    cur_pos = initial_submap.corners[0]
    overall_direction, initial_direction, rotation_required = self.getSweepDirection(initial_submap, cur_pos)
    cur_pos = self.refineSubmapStartpoint(cur_pos, overall_direction, initial_direction)

    if rotation_required:
      self.rotateBlock()

    allowable_submap_size = min(self.block_size_x, self.block_size_y)

    for i in ordered_submaps:
      # print('====== Submap ' + str(i) +  '=====')

      if submaps[i].size_x <= allowable_submap_size or submaps[i].size_y <= allowable_submap_size:
        continue

      movement = self.pathToNextSubmap(cur_pos, submaps[i])
      # print(cur_pos)
      # print(movement, i)
      path.extend(movement)
      cur_pos = Point(path[-1][0], path[-1][1])
      path.extend(self.lawnmower(cur_pos, submaps[i]))
      cur_pos = Point(path[-1][0], path[-1][1])
      # return path

    # movement = self.pathToNextSubmap(cur_pos, submaps[6])
    # print(movement)
    # path.extend(self.lawnmower(cur_pos, submaps[6]))

    return path
