from intervaltree import IntervalTree
import numpy as np
import math

from utils import Point, Pose, Direction

class SubmapPlanner:
  def __init__(self, occ_grid, block_size_x, block_size_y):

    self.occ_grid = occ_grid
    self.block_size_x = block_size_x
    self.block_size_y = block_size_y

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
    # TODO:
    # Check if block can fit in area -> block needs to be spun
    # Check if block can perfectly sweep area -> else block needs to be spun
    # Edge cases -> Block cant fit in area?


    max_block_dim = max(self.block_size_x, self.block_size_y)
    min_block_dim = min(self.block_size_x, self.block_size_y)

    max_area_dim = max(submap.size_x, submap.size_y)
    min_area_dim = min(submap.size_x, submap.size_y)

    if min_area_dim < max_block_dim:
      print("Blocked")

    general_direction = None
    start_direction = None

    #TODO: Create a function that correctly sets the general direction and start direction
    if max_area_dim == submap.size_x:
      general_direction = Direction.DOWN
      start_direction = Direction.RIGHT
    else:
      general_direction = Direction.RIGHT
      start_direction = Direction.DOWN

    rotate_90 = False

    if start_direction == Direction.RIGHT or start_direction == Direction.LEFT:
      rotate_90 = True

    # TODO: Cases for different start/general directions -> change start cell
    
    current_cell = None

    # Initial Position Calculations
    if general_direction == Direction.UP:
      current_cell = Point(submap.min_x + min_block_dim/2, submap.max_y - max_block_dim/2)
    elif general_direction == Direction.DOWN:
      current_cell = Point(submap.min_x + min_block_dim/2, submap.min_y + max_block_dim/2)
    elif general_direction == Direction.RIGHT:
      current_cell = Point(submap.min_x + max_block_dim/2, submap.min_y + min_block_dim/2)
    


    cell_path = [current_cell]

    sweeps = min_area_dim / max_block_dim
    moves_per_sweep = max_area_dim - min_block_dim + 1

    current_movement = start_direction

    # Boustrophedon method
    for i in range(int(sweeps)):
      if i % 2 == 0:
        current_movement = start_direction
      else:
        current_movement = start_direction.opposite()

      # Move along sweep line
      for j in range(int(moves_per_sweep)):
        current_cell = current_cell.shift(current_movement)
        cell_path.append(current_cell)

      # Break the loop so we dont perform the shift at the end
      if i == int(sweeps) - 1:
        continue
      
      # Shift to next sweep line
      for k in range(max_block_dim):
        current_cell = current_cell.shift[general_direction]
        cell_path.append(current_cell)

    if int(sweeps) != sweeps:
      remaining = min_area_dim - (int(sweeps) * max_block_dim) + 1

      # Need to move to next line and change direction if we are at the end of a sweep line
      if sweeps >= 1:

        current_movement = current_movement.opposite()

        # Shift to next sweep line
        for i in range(remaining):
          current_cell = current_cell.shift(general_direction)
          cell_path.append(current_cell) 

      # Move along sweep line
      for j in range(int(moves_per_sweep)):
        current_cell = current_cell.shift(current_movement)
        cell_path.append(current_cell)

    # Convert cell path to real world path
    path = []

    for c in cell_path:
      wx, wy = self.occ_grid.mapToWorld(c.x, c.y)

      if rotate_90:
        path.append(Pose(wx, wy, math.radians(90)))
      else:
        path.append(Pose(wx, wy, math.radians(0)))

    return path


  def process(self, submaps):
    self.createGraphFromSubmaps(submaps)

    # TODO: The optimal start and end vertex would be those that are attached to the two longest chains (so we dont need to back track them)
    reduced_adj_matrix = self.seperateLeafChains()

    hamiltonian_path_exists = self.heldKarp(reduced_adj_matrix)
    path = []

    if hamiltonian_path_exists:
      path = self.getHamiltonianPath(reduced_adj_matrix)
      
    path = list(map(lambda n: self.idx_map[n], path))

    for chain in self.leaf_chains:
      pos = path.index(chain[-1][1])
      
      for j in chain:
        path.insert(pos + 1, j[0])

    return path