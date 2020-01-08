from collections import deque

# Python implementation of https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm
class HopcroftKarp:

  def __init__(self, graph : dict):
    self.graph = graph

  def match(self):

    flatten = lambda x: [item for sublist in x for item in sublist]

    self.U = self.graph.keys()
    self.V = set(flatten(self.graph.values()))

    self.pair = {}
    self.dist = {}
    self.Q = deque()

    # Initialize pair and dist
    for u in self.U:
      self.pair[u] = None
      self.dist[u] = float('inf')

    for v in self.V:
      self.pair[v] = None
      self.dist[v] = float('inf')

    matching = 0

    while self.bfs():
      for u in self.U:
        if self.pair[u] is None:
          if self.dfs(u):
            matching = matching + 1

    return matching

  def bfs(self):
    for u in self.U:
      if self.pair[u] == None:
        self.dist[u] = 0
        self.Q.append(u)
      else:
        self.dist[u] = float('inf')

    self.dist[None] = float('inf')

    while len(self.Q) > 0:
      u = self.Q.popleft()

      if self.dist[u] < self.dist[None]:
        for v in self.graph[u]:
          if self.dist[self.pair[v]] == float('inf'):
            self.dist[self.pair[v]] = self.dist[u] + 1
            self.Q.append(self.pair[v])

    return self.dist[None] != float('inf')

  def dfs(self, u):
    if u is not None:
      for v in self.graph[u]:
        if self.dist[self.pair[v]] == self.dist[u] + 1:
          if self.dfs(self.pair[v]) == True:
            self.pair[v] = u
            self.pair[u] = v
            return True
      
      self.dist[u] = float('inf')
      return False
    return True