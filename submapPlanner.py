from intervaltree import IntervalTree
import numpy as np
import math

from utils import Point, Pose, Direction

class SubmapPlanner:
  def __init__(self, occ_grid, block_size_x, block_size_y):

    self.occ_grid = occ_grid
    self.block_size_x = block_size_x
    self.block_size_y = block_size_y

    self.cur_rot = 0

    self.adj_matrix = None
    self.dp_table = None

    self.leaf_chains = []
    self.idx_map = dict()  

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

    new_matrix = np.delete(self.adj_matrix, sorted_removed, 0)
    new_matrix = np.delete(new_matrix, sorted_removed, 1)

    self.idx_map = dict([(i, i) for i in range(new_matrix_size)])

    count = 0
    current = sorted_removed.pop()
    for i in range(new_matrix_size):
        if i == current and len(sorted_removed) > 0:
            count += 1
            current = sorted_removed.pop() - count

        self.idx_map[i] += count

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

      # TODO: An adjacency list implementation would allow this in constant time rather than linear in # of vertices
      neighbours = [idx for idx in range(len(reduced_adj_matrix)) if reduced_adj_matrix[current_vertex][idx] == 1]

      for nbr in neighbours:
        if nbr not in visited:
          if self.dp_table[nbr][table_bits^ 2**current_vertex]:
            table_bits = table_bits ^ 2**current_vertex
            current_vertex = nbr
            break

    # TODO: Remove
    if len(path) != len(reduced_adj_matrix):
      print("PROBLEM")
      exit()

    return path

  def generatePathForSubmap(self, submap):
    
    # TODO: Need to figure out initial position
    # TODO: Need to calculate overall and initial direction

    cur_pos = (submap.min_x, submap.min_y)

    cell_path = self.lawnmower(cur_pos, submap.size_x, submap.size_y, )

    rotate_90 = False

    # Convert cell path to real world path
    path = []

    for c in cell_path:
      wx, wy = self.occ_grid.mapToWorld(c.x, c.y)

      if rotate_90:
        path.append(Pose(wx, wy, math.radians(90)))
      else:
        path.append(Pose(wx, wy, math.radians(0)))

    return path


  def lawnmower(self, start_point, submap):
    overall_direction, initial_direction, rotation_required = self.getSweepDirection(submap, start_point)

    if rotation_required:
      prev_x = self.block_size_x
      prev_y = self.block_size_y
      self.block_size_x = prev_y
      self.block_size_y = prev_x
      
      # Since we are rotating, we need to shift the start point accordingly
      if overall_direction == Direction.UP:
        if initial_direction == Direction.LEFT:
          start_point.x += prev_x//2 - self.block_size_x//2
          start_point.y += prev_y//2 - self.block_size_y//2
        else:
          start_point.x += -prev_x//2 + self.block_size_x//2
          start_point.y += prev_y//2 - self.block_size_y//2
      elif overall_direction == Direction.DOWN:
        if initial_direction == Direction.LEFT:
          start_point.x += prev_x//2 - self.block_size_x//2
          start_point.y += -prev_y//2 + self.block_size_y//2
        else:
          start_point.x += -prev_x//2 + self.block_size_x//2
          start_point.y += -prev_y//2 + self.block_size_y//2
      elif overall_direction == Direction.LEFT:
        if initial_direction == Direction.UP:
          start_point.x += prev_x//2 - self.block_size_x//2
          start_point.y += prev_y//2 - self.block_size_y//2
        else:
          start_point.x += prev_x//2 - self.block_size_x//2
          start_point.y += -prev_y//2 + self.block_size_y//2
      else:
        if initial_direction == Direction.UP:
          start_point.x += -prev_x//2 + self.block_size_x//2
          start_point.y += prev_y//2 - self.block_size_y//2
        else:
          start_point.x += -prev_x//2 + self.block_size_x//2
          start_point.y += -prev_y//2 + self.block_size_y//2


      # Set to 90 deg if 0 or set to 0 otherwise
      if self.cur_rot == 0:
        self.cur_rot = math.radians(90)
      else:
        self.cur_rot = 0
      
    path = [Pose(start_point.x, start_point.y, self.cur_rot)]
    
    sweep_length = 0
    number_of_sweeps = 0
    remaining = 0

    temp_pos = start_point

    # overall direction and initial direction must be perpindicular
    assert overall_direction != initial_direction
    assert overall_direction.opposite() != initial_direction

    print(self.block_size_x, self.block_size_y, initial_direction, overall_direction)
    
    sweep_length = max(submap.size_x, submap.size_y) - min(self.block_size_x, self.block_size_y)
    number_of_sweeps = min(submap.size_x, submap.size_y) // max(self.block_size_x, self.block_size_y)
    remaining = min(submap.size_x, submap.size_y) % max(self.block_size_x, self.block_size_y)

    for count, n in enumerate(range(number_of_sweeps)):

      # Move along sweep line
      for i in range(sweep_length):
        temp_pos = temp_pos.shift(initial_direction)
        pose = Pose(temp_pos.x, temp_pos.y, self.cur_rot)
        path.append(pose)

      initial_direction = initial_direction.opposite()

      # Exit here on last movement (dont shift when we reach end)
      if count == number_of_sweeps - 1:
        break

      # Shift to next sweep line
      for j in range(max(self.block_size_x, self.block_size_y)):
        temp_pos = temp_pos.shift(overall_direction)
        pose = Pose(temp_pos.x, temp_pos.y, self.cur_rot)
        path.append(pose)

    # This will cover the case where the rectangle is not a perfect division of the length of the sweeper
    # NOTE: This causes double coverage along the entire sweep line
    if remaining != 0:
      # Shift to next sweep line
      for j in range(remaining):
        temp_pos = temp_pos.shift(overall_direction)
        pose = Pose(temp_pos.x, temp_pos.y, self.cur_rot)
        path.append(pose)
      
      for i in range(sweep_length):
        temp_pos = temp_pos.shift(initial_direction)
        pose = Pose(temp_pos.x, temp_pos.y, self.cur_rot)
        path.append(pose)

    return path

  def getSweepDirection(self, submap, start_point):

    # TODO: Edge case occurs when we rotate the block but the block is too wide to do so

    rotation_required = False
    overall_direction = None
    initial_direction = None

    diff_to_top = start_point.y - submap.min_y
    diff_to_bot = submap.max_y - start_point.y

    diff_to_left = start_point.x - submap.min_x
    diff_to_right = submap.max_x - start_point.x

    assert diff_to_top > 0
    assert diff_to_bot > 0
    assert diff_to_left > 0
    assert diff_to_right > 0


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

    return (overall_direction, initial_direction, rotation_required)

  def moveToNextSubmap(self, cur_pos, next_submap):
    target_corner = next_submap.corners[0]
    min_distance = cur_pos.distance_to(target_corner)

    for i in next_submap.corners[:-1]:
      cur_dist = cur_pos.distance_to(i)

      if cur_dist < min_distance:
        target_corner = i
    
    # Find path to target corner


  def process(self, submaps):
    self.createGraphFromSubmaps(submaps)

    # TODO: The optimal start and end vertex would be those that are attached to the two longest chains (so we dont need to back track them)
    reduced_adj_matrix = self.seperateLeafChains()

    hamiltonian_path_exists = self.heldKarp(reduced_adj_matrix)
    ordered_submaps = []

    if hamiltonian_path_exists:
      ordered_submaps = self.getHamiltonianPath(reduced_adj_matrix)
      
    ordered_submaps = list(map(lambda n: self.idx_map[n], ordered_submaps))

    # for chain in self.leaf_chains:
    #   pos = ordered_submaps.index(chain[-1][1])
      
    #   for j in chain:
    #     ordered_submaps.insert(pos + 1, j[0])

    path = []

    print(ordered_submaps)
    print(self.leaf_chains)

    path = []
    # ordered submaps contains indexes of submaps
    # for cur in ordered_submaps:
      # path += self.lawnmower(submaps[cur])
    # path = self.lawnmower(submaps[13].corners[0] + Point(self.block_size_x//2, -self.block_size_y//2), submaps[13])

    return path