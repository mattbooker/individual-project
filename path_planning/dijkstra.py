import occupancyGrid
import heapq as h

class Dijkstra:

  def __init__(self, occ_grid, start_pose, target_pose):
    self.occ_grid = occ_grid
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
    visited = set()
    prev = dict()
    # distance = dict()
    pq = []

    prev[(self.start_mx, self.start_my)] = (0, None)

    # prev[(self.start_mx, self.start_my)] = None
    # distance[(self.start_mx, self.start_my)] = 0

    # h.heappush(pq, self.start_mx, self.start_my))
    h.heappush(pq, (0, (self.start_mx, self.start_my)))

    while len(pq) != 0:
      # current_mx, current_my = h.heappop(pq)
      current_cost, (current_mx, current_my) = h.heappop(pq)

      # If not visited then add to visited. Otherwise skip 
      # (we check here as well because we could have added a node twice before visiting it)
      if (current_mx, current_my) not in visited:
        visited.add((current_mx, current_my))
      else:
        continue

      # If we found the goal then return path
      if current_mx == self.goal_mx and current_my == self.goal_my:
        return self.get_path(prev)

      # Otherwise insert all neighbouring points, unless visited
      for mx, my in self.nhood4(current_mx, current_my):
        # if (mx, my) not in visited:
        # alternate_dist = distance to current + distance between current and neighbour

          # if alternate_dist < distance to neighbour:
            # distance to neighbour = alt
            # previous of neighbour = current

        if (mx, my) not in visited:
          if (mx, my) in prev and (current_cost + 1) < prev[(mx, my)][0]:
            prev[(mx, my)] = (current_cost + 1, (current_mx, current_my))
          else: 
            prev[(mx, my)] = (current_cost + 1, (current_mx, current_my))

          h.heappush(pq, (current_cost+1, (mx, my)))

  def get_path(self, prev : dict):
    result = [(self.goal_mx, self.goal_my)]

    try:
      _, (current_mx, current_my) = prev[(self.goal_mx, self.goal_my)]
    except KeyError:
      print(prev.items())

    while prev[(current_mx, current_my)][1] != None:
      result.append(self.occ_grid.map_to_world(current_mx, current_my))

      try:
        _, (current_mx, current_my) = prev[(current_mx, current_my)]
      except KeyError:
        print(prev.items())

    return result[::-1]