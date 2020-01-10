from collections import deque


class MaximumIndependentSet:

  def __init__(self, bipartite_graph : dict):
    self.bipartite_graph = bipartite_graph

  def compute(self):
    # Hopcroft-Karp gets the maximal matching of the bipartite graph in O(E sqrt(V))
    self.hopcrofKarp()

    # Using Koenigs theorem we can retreive the minimum vertex cover in O(V)
    self.koenig()

    self.max_independent_set = set()

    for vertex in self.U:
      if vertex not in self.min_vertex_cover:
        self.max_independent_set.add(vertex)
    
    for vertex in self.V:
      if vertex not in self.min_vertex_cover:
        self.max_independent_set.add(vertex)
    

  # Python implementation of Hopcroft-Karp Algorithm from https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm
  def hopcrofKarp(self):

    flatten = lambda x: [item for sublist in x for item in sublist]

    self.U = self.bipartite_graph.keys()
    self.V = set(flatten(self.bipartite_graph.values()))

    self.pairs = {}
    self.dist = {}
    self.Q = deque()

    # Initialize pairs and dist
    for u in self.U:
      self.pairs[u] = None
      self.dist[u] = float('inf')

    for v in self.V:
      self.pairs[v] = None
      self.dist[v] = float('inf')

    matching = 0

    while self.bfs():
      for u in self.U:
        if self.pairs[u] is None:
          if self.dfs(u):
            matching = matching + 1

    return matching

  # Used in HopcroftKarp
  def bfs(self):
    for u in self.U:
      if self.pairs[u] == None:
        self.dist[u] = 0
        self.Q.append(u)
      else:
        self.dist[u] = float('inf')

    self.dist[None] = float('inf')

    while len(self.Q) > 0:
      u = self.Q.popleft()

      if self.dist[u] < self.dist[None]:
        for v in self.bipartite_graph[u]:
          if self.dist[self.pairs[v]] == float('inf'):
            self.dist[self.pairs[v]] = self.dist[u] + 1
            self.Q.append(self.pairs[v])

    return self.dist[None] != float('inf')

  # Used in HopcroftKarp
  def dfs(self, u):
    if u is not None:
      for v in self.bipartite_graph[u]:
        if self.dist[self.pairs[v]] == self.dist[u] + 1:
          if self.dfs(self.pairs[v]) == True:
            self.pairs[v] = u
            self.pairs[u] = v
            return True
      
      self.dist[u] = float('inf')
      return False
    return True

  # Use Koenigs theorem to get the minimum vertex cover of the bipartite_graph
  def koenig(self):
    unmatched_left = set()
    alternating_connected = set()
    bfs = []

    for vertex in self.U:
      if self.pairs[vertex] is None:
        unmatched_left.add(vertex)

        for nbr in self.bipartite_graph[vertex]:
          bfs.append((nbr, 0))

        while len(bfs) != 0:
          cur_vertex, edge_type = bfs.pop()
          alternating_connected.add(cur_vertex)

          # Added via an unmatched edge therefore need matched
          if edge_type == 0:
            bfs.append((self.pairs[cur_vertex], 1))
          
          # Otherwise added via a matched edge so add all except the edge we came from and any visited already
          else:
            for nbr in self.bipartite_graph[cur_vertex]:
              if nbr not in alternating_connected and self.pairs[cur_vertex] != cur_vertex:
                bfs.append((nbr, 0))

    alternating_connected = alternating_connected.union(unmatched_left)
    self.min_vertex_cover = (set(self.U) - alternating_connected).union(set(self.V).intersection(alternating_connected))