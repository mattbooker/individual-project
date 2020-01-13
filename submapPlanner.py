from intervaltree import IntervalTree
import numpy as np

class SubmapPlanner:
  def __init__(self):
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