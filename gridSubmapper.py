from collections import defaultdict

import cv2
import numpy as np

from maximumIndependentSet import MaximumIndependentSet
from occupancyGrid import OccupancyGrid
from utils import Direction, Point, Submap


class GridSubmapper:

  def __init__(self, occ_grid):
    self.occ_grid = occ_grid.clone()
    self.rectilinear_occ_grid = occ_grid.generateRectilinearOcc()
    self.definite_rectangles = []
    self.possible_rectangles = set()
    self.submaps = []

  def getNhoodOccupancy(self, x, y):
    '''
    Returns a bool array with 4 elements corresponding to whether the top, left, bottom or right cells are occupied.
    Out of bounds is treated as unoccupied.
    '''

    # Top, Left, Bottom, Right
    occupied = [False, False, False, False]

    # Check Top
    if y - 1 >= 0 and self.rectilinear_occ_grid[x, y - 1] == 1:
      occupied[0] = True

    # Check Left
    if x - 1 >= 0 and self.rectilinear_occ_grid[x - 1, y] == 1:
      occupied[1] = True

    # Check Bottom
    if y + 1 < self.rectilinear_occ_grid.size_y and self.rectilinear_occ_grid[x, y + 1] == 1:
      occupied[2] = True

    # Check Right
    if x + 1 < self.rectilinear_occ_grid.size_x and self.rectilinear_occ_grid[x + 1, y] == 1:
      occupied[3] = True

    return occupied

  def getNhoodValues(self, x, y):
    '''
    Returns an array containing the values above, left, below and right of the given x,y.
    Out of bounds is treated as 1.
    '''

    # Top, Left, Bottom, Right
    occupied = [1, 1, 1, 1]

    # Check Top
    if y - 1 >= 0:
      occupied[0] = self.rectilinear_occ_grid[x, y - 1]

    # Check Left
    if x - 1 >= 0:
      occupied[1] = self.rectilinear_occ_grid[x - 1, y]

    # Check Bottom
    if y + 1 < self.rectilinear_occ_grid.size_y:
      occupied[2] = self.rectilinear_occ_grid[x, y + 1]

    # Check Right
    if x + 1 < self.rectilinear_occ_grid.size_x:
      occupied[3] = self.rectilinear_occ_grid[x + 1, y]

    return occupied

  def isCorner(self, x, y):
    # If point is not occupied then it cant be a corner
    if self.rectilinear_occ_grid[x, y] == 0:
      return False

    on_boundary = True if (x == 0 or x == self.rectilinear_occ_grid.size_x - 1 or y == 0 or y == self.rectilinear_occ_grid.size_y - 1) else False

    occupied = self.getNhoodOccupancy(x, y)

    # If we are on the boundary then a corner only forms if a surrounding 4-neighbourhood has 1 occupied
    if on_boundary:
      if sum(occupied) == 1:
        return True
    elif sum(occupied) == 2:  # Otherwise a corner only forms when the 4-nhood has 2 occupied
      # Check the case where the two occupied positions form a straight line
      if (occupied[0] and occupied[2]) or (occupied[1] and occupied[3]):
        return False
      else:
        return True

    return False

  def getCorners(self):
    corners = []

    for j in range(self.rectilinear_occ_grid.size_y):
      for i in range(self.rectilinear_occ_grid.size_x):
        if self.isCorner(i, j):
          corners.append(Point(i,j))
    
    return corners

  def seperateCorners(self, corners):
    '''
    Returns all pairs of points in corners that are cogrid, i.e. that lie along the same x or y line. The result
    is split into two lists vertical pairs and horizontal pairs.
    '''

    # Naive O(n^2) implementation
    # Could be improved via sorting
    has_cogrid_pair = [False] * len(corners)

    vertical_pairs = []
    horizontal_pairs = []

    # Iterate through all points to find cogrid vertices. Need to ensure a clear line exists between the two points
    for a in range(len(corners) - 1):
      for b in range(a + 1, len(corners)):
        if corners[a].x == corners[b].x:

          # Check there is an uniterrupted line between the two points
          start = min(corners[a].y, corners[b].y)
          end = max(corners[a].y, corners[b].y)

          isValid = True

          for idx in range(start + 1, end):
            if self.rectilinear_occ_grid[corners[a].x, idx] == 1:
              isValid = False
              break
          
          # If the two points form a valid cogrid pair then add to our result
          if isValid:
            vertical_pairs.append((corners[a], corners[b]))
            has_cogrid_pair[a] = True
            has_cogrid_pair[b] = True
          
        elif corners[a].y == corners[b].y:

          # Check there is an uniterrupted line between the two points
          start = min(corners[a].x, corners[b].x)
          end = max(corners[a].x, corners[b].x)

          isValid = True

          for idx in range(start + 1, end):
            if self.rectilinear_occ_grid[idx, corners[a].y] == 1:
              isValid = False
              break

          # If the two points form a valid cogrid pair then add to our result
          if isValid:
            horizontal_pairs.append((corners[a], corners[b]))

            has_cogrid_pair[a] = True
            has_cogrid_pair[b] = True

    remaining_corners = [corners[i] for i in range(len(corners)) if has_cogrid_pair[i] == False]

    return vertical_pairs, horizontal_pairs, remaining_corners

  def getEdgeDirection(self, x, y):

    # [top, left, bottom, right]
    occupied = self.getNhoodOccupancy(x, y)

    # Take first free direction rather than random dir
    direction = occupied.index(False)

    # Check what to mark the cell as based on whats around it
    check_idx = (direction - 1) % 4

    if occupied[check_idx]:
      if check_idx == 0:
        return Direction.LEFT
        
      elif check_idx == 1:
        return Direction.DOWN

      elif check_idx == 3:
        return Direction.UP

      else:
        # The case where check_idx is 2 should never occur because we always choose the first unoccupied in CCW direction
        print("ERROR")
        exit() # TODO:

    # An edge case occurs when the cells above the corner and to the right of the corner are unoccupied
    elif occupied[(direction + 1) % 4]:
      return Direction.SPECIAL

  def markEdge(self, x, y, direction):

    cur_point = Point(x, y)

    if direction == Direction.SPECIAL:

      # Move upwards
      direction = Direction.UP
      cur_point = cur_point.shift(direction)

      # Occurs if another edge runs over this one
      if self.rectilinear_occ_grid[cur_point.x, cur_point.y] < 0:
        return False

      while self.rectilinear_occ_grid.inBounds(cur_point.x, cur_point.y) and self.rectilinear_occ_grid[cur_point.x, cur_point.y] == 0:
        # But mark the cell with down
        self.rectilinear_occ_grid[cur_point.x, cur_point.y] = direction.opposite().value
        cur_point = cur_point.shift(direction)

      if self.rectilinear_occ_grid.inBounds(cur_point.x, cur_point.y) and self.rectilinear_occ_grid[cur_point.x, cur_point.y] == direction.next().value:
        self.rectilinear_occ_grid[cur_point.x, cur_point.y] = Direction.INTERSECTION.value

    else:
      cur_point = cur_point.shift(direction)

      # Occurs if another edge runs over this one
      if self.rectilinear_occ_grid[cur_point.x, cur_point.y] < 0:
        return False

      while self.rectilinear_occ_grid.inBounds(cur_point.x, cur_point.y) and self.rectilinear_occ_grid[cur_point.x, cur_point.y] == 0:
        self.rectilinear_occ_grid[cur_point.x, cur_point.y] = direction.value
        cur_point = cur_point.shift(direction)

      # TODO:
      if self.rectilinear_occ_grid.inBounds(cur_point.x, cur_point.y) and self.rectilinear_occ_grid[cur_point.x, cur_point.y] == direction.next().value:
        self.rectilinear_occ_grid[cur_point.x, cur_point.y] = Direction.INTERSECTION.value

    return True

  def markCogrid(self, x1, y1, x2, y2, vertical_line):
    occupied_1 = self.getNhoodOccupancy(x1, y1)
    occupied_2 = self.getNhoodOccupancy(x2, y2)

    # Right rotate occupied_1
    right_rotate = occupied_1[-1:] + occupied_1[:-1]

    # Left rotate occupied_1
    left_rotate = occupied_1[1:] + occupied_1[:1]

    if right_rotate == occupied_2:
      if vertical_line:
        # Start from top most point and move down
        start = Point(x1, min(y1, y2))
        self.markEdge(start.x, start.y, Direction.DOWN)

        # Return the start point shifted one (so that it is the corner of the rectangle) as well as an adjacent tile (that forms a potential rectangle)
        rect_corner = start.shift(Direction.DOWN)
        return [rect_corner], [rect_corner.shift(Direction.RIGHT)]

      else:
        # Start from left most point and move right
        start = Point(min(x1, x2), y1)
        self.markEdge(start.x, start.y, Direction.RIGHT)

        rect_corner = start.shift(Direction.RIGHT)
        return [rect_corner], [rect_corner.shift(Direction.UP)]
    
    elif left_rotate == occupied_2:
      if vertical_line:
        # Start from bottom most point and move up
        start = Point(x1, max(y1, y2))
        self.markEdge(start.x, start.y, Direction.UP)

        rect_corner = start.shift(Direction.UP)
        return [rect_corner], [rect_corner.shift(Direction.LEFT)]

      else:
        # Start from right most point and move left
        start = Point(max(x1, x2), y1)
        self.markEdge(start.x, start.y, Direction.LEFT)


        rect_corner = start.shift(Direction.LEFT)
        return [rect_corner], [rect_corner.shift(Direction.DOWN)]

    
    elif occupied_1 == occupied_2[::-1]:
      '''
      If we reach here it means we have two corners whose leading edge move in opposite directions i.e.
          |                     |
        --+     +--   or      --+
                |                     
                                +--
                                |
      '''

      if vertical_line:
        downward_start_point = Point(x1 - 1, min(y1,y2))
        upward_start_point = Point(x1 + 1, max(y1, y2))

        self.markEdge(downward_start_point.x, downward_start_point.y, Direction.DOWN)
        self.markEdge(upward_start_point.x, upward_start_point.y, Direction.UP)

        # TODO: Include the 1 line that forms between rectangles in possible_rect?
        return [downward_start_point.shift(Direction.DOWN), upward_start_point.shift(Direction.UP)], []

      else:      
        # For this case with horizontal lines, the edge ends on the corner rather than beginning at it thus we need to run markEdge backwards
        leftward_end_point = Point(min(x1, x2), y1 - 1)
        rightward_end_point = Point(max(x1, x2), y1 + 1)

        # Get the start points of the edge (using the end_point will double add a corner during the rectangle phase)
        result = []

        cur_point = leftward_end_point.shift(Direction.RIGHT)
        while self.rectilinear_occ_grid.inBounds(cur_point.x, cur_point.y) and self.rectilinear_occ_grid[cur_point.x, cur_point.y] == 0:
          self.rectilinear_occ_grid[cur_point.x, cur_point.y] = Direction.LEFT.value
          cur_point = cur_point.shift(Direction.RIGHT)

        result.append(cur_point.shift(Direction.LEFT))

        cur_point = rightward_end_point.shift(Direction.LEFT)
        while self.rectilinear_occ_grid.inBounds(cur_point.x, cur_point.y) and self.rectilinear_occ_grid[cur_point.x, cur_point.y] == 0:
          self.rectilinear_occ_grid[cur_point.x, cur_point.y] = Direction.RIGHT.value
          cur_point = cur_point.shift(Direction.LEFT)

        result.append(cur_point.shift(Direction.RIGHT))

        return result, []

    else:
      '''
      Same case as above except the edges are flipped i.e.
                |             |
        --+     +--   or      +--
          |                          
                            --+
                              |
      '''

      if vertical_line:
        # For this case with vertical lines, the edge ends on the corner rather than beginning at it thus we need to run the mark edge backwards
        upward_end_point = Point(x1 + 1, min(y1, y2))
        downward_end_point = Point(x1 - 1, max(y1, y2))

        # Get the start points of the edge (using the end_point will double add a corner during the rectangle phase)
        result = []

        cur_point = upward_end_point.shift(Direction.DOWN)
        while self.rectilinear_occ_grid.inBounds(cur_point.x, cur_point.y) and self.rectilinear_occ_grid[cur_point.x, cur_point.y] == 0:
          self.rectilinear_occ_grid[cur_point.x, cur_point.y] = Direction.UP.value
          cur_point = cur_point.shift(Direction.DOWN)

        result.append(cur_point.shift(Direction.UP))

        cur_point = downward_end_point.shift(Direction.UP)
        while self.rectilinear_occ_grid.inBounds(cur_point.x, cur_point.y) and self.rectilinear_occ_grid[cur_point.x, cur_point.y] == 0:
          self.rectilinear_occ_grid[cur_point.x, cur_point.y] = Direction.DOWN.value
          cur_point = cur_point.shift(Direction.UP)

        result.append(cur_point.shift(Direction.DOWN))

        return result, []

      else:
        leftward_initial_point = Point(max(x1, x2), y1 - 1)
        rightward_initial_point = Point(min(x1, x2), y1 + 1)

        self.markEdge(leftward_initial_point.x, leftward_initial_point.y, Direction.LEFT)
        self.markEdge(rightward_initial_point.x, rightward_initial_point.y, Direction.RIGHT)

        return [leftward_initial_point.shift(Direction.LEFT), rightward_initial_point.shift(Direction.RIGHT)], []

  def makeRectangle(self, x, y, possible_rectangles):
    '''

    There is an edge case that occurs when an UP edge is broken in two by a LEFT edge. However this is a non-issue since the 
    order in which we search for edges and mark them is from top to bottom from left to right. This means the LEFT edge will always
    occur first before an upwards edge, i.e. only a LEFT edge will be split by an UP edge not vice-versa.
    '''

    initial_point = Point(x,y)

    cur_dir = Direction(self.rectilinear_occ_grid[x, y])
    cur_point = initial_point
    cur_point = cur_point.shift(cur_dir)

    rectangle_corners = [initial_point]

    cell_value = self.rectilinear_occ_grid[cur_point.x, cur_point.y]

    # Check the edge case where we start on a special corner
    if cur_dir == Direction.DOWN and self.rectilinear_occ_grid[cur_point.x, cur_point.y] == 1:
      cur_dir = Direction.LEFT
      cur_point = initial_point
      cur_point = cur_point.shift(cur_dir)

    while cur_point != initial_point:

      # Update the possible rectangles
      if cur_point in possible_rectangles:
        possible_rectangles.remove(cur_point)

      if len(rectangle_corners) > 4:
        print("PROBLEM")
        print(rectangle_corners)
        exit()

      # Check that the current point is in bounds
      if self.rectilinear_occ_grid.inBounds(cur_point.x, cur_point.y):
        cell_value = self.rectilinear_occ_grid[cur_point.x, cur_point.y]

      # Otherwise backtrack, change direction and mark the point as a corner in the rectangle
      else:
        cur_point = cur_point.shift(cur_dir.opposite())
        cur_dir = cur_dir.next()
        rectangle_corners.append(cur_point)

        # Move in new direction so that we dont have issues with the error checking steps below
        cur_point = cur_point.shift(cur_dir)

        # Go to next iteration
        continue
      
      # print(cur_point, end="--- ")
      # print(cell_value)

      # If we hit a cell that has 0 or the current direction then we continue moving in same direction
      if cell_value == 0 or cell_value == cur_dir.value:
        cur_point = cur_point.shift(cur_dir)

      # If we encounter an a cell with the next direction then add the corner and follow the next direction
      elif cell_value == cur_dir.next().value:
        # Move with new direction
        cur_dir = cur_dir.next()
        rectangle_corners.append(cur_point)

        cur_point = cur_point.shift(cur_dir)

      # If we encounter an intersection check the value in the next direction after the intersection.
      elif cell_value == Direction.INTERSECTION.value:
        cur_dir = cur_dir.next()
        potential_corner = cur_point
        cur_point = cur_point.shift(cur_dir)

        if self.rectilinear_occ_grid[cur_point.x, cur_point.y] == cur_dir.value:
          rectangle_corners.append(potential_corner)

        else:
          potential_corner = potential_corner.shift(cur_dir.next())
          rectangle_corners.append(potential_corner)
          cur_point = potential_corner.shift(cur_dir)


      # If we hit an obstacle (i.e. 1) or other marked cell then backtrack, change direction and mark point as a corner in the rectangle
      else:
        cur_point = cur_point.shift(cur_dir.opposite())
        cur_dir = cur_dir.next()

        rectangle_corners.append(cur_point)

        # Move in new direction so that we dont have issues with the error checking steps
        cur_point = cur_point.shift(cur_dir)
    
    return Submap(rectangle_corners)

  def splitIntoRectangles(self, concave_corners):
    '''
    Given a list of concave corner points, splits the rectilinear_occ_grid into rectangles. Returns a list of points that are the corners of unique rectangles and
    a set of points that lie on the edges of other potential rectangles that were missed. For any split it can create either 1 or 2 rectangles, thus
    the set is used to keep track of extra rectangles.
    '''
    
    definite_rectangles = []
    possible_rectangles = set()

    for corner in concave_corners:
      direction = self.getEdgeDirection(corner.x, corner.y)
      suitable_edge = self.markEdge(corner.x, corner.y, direction)

      if not suitable_edge:
        continue

      # Based on the direction of the edge add the corner of the rectangle as well as the adjacent rectangle
      if direction == Direction.SPECIAL:
        definite_rectangles.append(corner.shift(Direction.UP))

        # Offset contains the previous shift
        possible_rectangles.add(corner.shift(Direction.RIGHT))
      else:
        definite_rectangles.append(corner.shift(direction))

        # Offset contains the previous shift
        possible_rectangles.add(corner.shift(direction.next().opposite()))

    return definite_rectangles, possible_rectangles

  def extractSubmaps(self):
    used_corners = set()

    # STEP 1: Iterate through the points known to be rectangle corners
    for corner in self.definite_rectangles:

      # Skip this corner if it has been used in another rectangle
      if corner in used_corners:
        continue
      
      submap = self.makeRectangle(corner.x, corner.y, self.possible_rectangles)

      # Add the corners of this submap into the used corners set
      for p in submap.corners:
        used_corners.add(p)

      self.submaps.append(submap)


    # STEP 2: Iterate through the rectangles that may have not been accounted for
    while len(self.possible_rectangles) > 0:
      cell = self.possible_rectangles.pop()

      occupied = self.getNhoodValues(cell.x, cell.y)

      direction_of_corner = None

      if occupied.count(0) == 3:
        if occupied[0]:
          direction_of_corner = Direction.RIGHT
        elif occupied[1]:
          direction_of_corner = Direction.UP
        elif occupied[2]:
          direction_of_corner = Direction.LEFT
        elif occupied[3]:
          direction_of_corner = Direction.DOWN

        # Need to move to the closest corner
        while occupied.count(0) != 2:
          cell = cell.shift(direction_of_corner)
          occupied = self.getNhoodValues(cell.x, cell.y)

        direction_of_corner = direction_of_corner.next()

      else:
        if occupied[3] and occupied[0]:
          direction_of_corner = Direction.DOWN
        elif occupied[0] and occupied[1]:
          direction_of_corner = Direction.RIGHT
        elif occupied[1] and occupied[2]:
          direction_of_corner = Direction.UP
        elif occupied[2] and occupied[3]:
          direction_of_corner = Direction.LEFT
      
      if direction_of_corner == None:
        # print(occupied)
        continue

      # Set the cell to have the correct direction and then make a submap
      self.rectilinear_occ_grid[cell.x, cell.y] = direction_of_corner.value
      self.submaps.append(self.makeRectangle(cell.x, cell.y, self.possible_rectangles))

  def handleCogridCorners(self, vertical_pairs, horizontal_pairs):
    bipartite_graph = defaultdict(list)

    # Used to keep track of points that dont make it into the graph
    isolated_vertical = []
    isolated_horizontal = []
    
    # Create a temp_rectilinear_occ_grid to use for faster construction of the bipartite graph (if we use rectilinear_occ_grid it will leave remnants which we dont want)
    temp_rectilinear_occ_grid = self.rectilinear_occ_grid.clone()

    # Mark the vertical cogrid lines
    for num, (point_1, point_2) in enumerate(vertical_pairs, 2):

      # Find start and end points
      start_y = min(point_1.y, point_2.y)
      end_y = max(point_1.y, point_2.y)

      # Mark the cells from start to end with num
      for i in range(start_y, end_y + 1):
        temp_rectilinear_occ_grid[point_1.x, i] = num
      
    # Mark horizontal cogrid lines and build the bipartite graph
    for num, (point_1, point_2) in enumerate(horizontal_pairs, 2 + len(vertical_pairs)):
      
      independent_node = True

      # Find start and end poitns
      start_x = min(point_1.x, point_2.x)
      end_x = max(point_1.x, point_2.x)

      for i in range(start_x, end_x + 1):
        val = int(temp_rectilinear_occ_grid[i, point_1.y])

        if val > 1:
          independent_node = False
          bipartite_graph[val].append(num)

        temp_rectilinear_occ_grid[i, point_1.y] = num

      # Keep track of the point if it is not part of the graph
      if independent_node:
        isolated_horizontal.append((point_1, point_2))

    # Find the vertical pairs that didnt make it into the graph
    for num, pair in enumerate(vertical_pairs, 2):
      if num not in bipartite_graph:
        isolated_vertical.append(pair)

    MIS = MaximumIndependentSet(bipartite_graph)
    MIS.compute()

    # Maintain a set of used points
    used_points = set()

    # Mark the corners that are part of the MIS
    for num in MIS.max_independent_set:
      idx = num - 2

      cogrid_definite_rect_corners = None
      cogrid_possible_rect_corners = None

      if idx < len(vertical_pairs):
        point_1, point_2 = vertical_pairs[idx]
        cogrid_definite_rect_corners, cogrid_possible_rect_corners = self.markCogrid(point_1.x, point_1.y, point_2.x, point_2.y, True)

        used_points.add(point_1)
        used_points.add(point_2)

      else:
        idx -= len(vertical_pairs)
        point_1, point_2 = horizontal_pairs[idx]
        cogrid_definite_rect_corners, cogrid_possible_rect_corners = self.markCogrid(point_1.x, point_1.y, point_2.x, point_2.y, False)

        used_points.add(point_1)
        used_points.add(point_2)

      self.definite_rectangles.extend(cogrid_definite_rect_corners)
      self.possible_rectangles.update(cogrid_possible_rect_corners)

    # Mark cogrid corners that were not included in the bipartite graph
    for (point_1, point_2) in isolated_vertical:
      cogrid_definite_rect_corners, cogrid_possible_rect_corners = self.markCogrid(point_1.x, point_1.y, point_2.x, point_2.y, True)

      self.definite_rectangles.extend(cogrid_definite_rect_corners)
      self.possible_rectangles.update(cogrid_possible_rect_corners)

    for (point_1, point_2) in isolated_horizontal:
      cogrid_definite_rect_corners, cogrid_possible_rect_corners = self.markCogrid(point_1.x, point_1.y, point_2.x, point_2.y, False)
      
      self.definite_rectangles.extend(cogrid_definite_rect_corners)
      self.possible_rectangles.update(cogrid_possible_rect_corners) 

    # Lastly, find the leftover points that are not part of the MIS so we use them for further decomposition later
    leftover_corners = []
    
    for num in MIS.min_vertex_cover:
      idx = num - 2

      if idx < len(vertical_pairs):
        point_1, point_2 = vertical_pairs[idx]
        
        if point_1 not in used_points:
          leftover_corners.append(point_1)
        
        if point_2 not in used_points:
          leftover_corners.append(point_2)

      else:
        idx -= len(vertical_pairs)
        point_1, point_2 = horizontal_pairs[idx]

        if point_1 not in used_points:
          leftover_corners.append(point_1)
        
        if point_2 not in used_points:
          leftover_corners.append(point_2)

    return leftover_corners

  def handleRemainingCorners(self, noncogrid_corners, leftover_corners):
    corners = noncogrid_corners + leftover_corners

    definite_remaining_rectangles, possible_remaining_rectangles = self.splitIntoRectangles(corners)

    self.definite_rectangles.extend(definite_remaining_rectangles)
    self.possible_rectangles.update(possible_remaining_rectangles)

  def handleNonRectangularRegions(self):
    difference = self.rectilinear_occ_grid.grid - self.occ_grid.grid
    difference[difference <= 0] = 0
    difference = np.uint8(difference)

    contours, _ = cv2.findContours(difference, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for shape in contours:

      # Convert the numpy array to list of tuples
      points = [Point(x,y) for x, y in shape.reshape(len(shape), 2)]
    
      non_rectangle_submap = Submap(points, False)

      self.submaps.append(non_rectangle_submap)
      
  def visualization(self):
    visual_grid = self.rectilinear_occ_grid.clone()
    visual_grid.grid[visual_grid.grid > 0] = -1
    visual_grid.grid[visual_grid.grid < 0] = 0

    for num, submap in enumerate(self.submaps):
      for (x, y) in submap.range():
        visual_grid[x, y] = num

    return visual_grid

  def process(self, block_size_x, block_size_y):
    # Find the concave corners
    concave_corners = self.getCorners()

    # If we found corners then process them
    if len(concave_corners) != 0:

      # Preproces corners
      vertical_cogrid, horizontal_cogrid, noncogrid_corners = self.seperateCorners(concave_corners)

      leftover_corners = self.handleCogridCorners(vertical_cogrid, horizontal_cogrid)

      self.handleRemainingCorners(noncogrid_corners, leftover_corners)

      # Extract
      self.extractSubmaps()

      # Need to handle the remaining non rectangular regions
      self.handleNonRectangularRegions()
    
    # If no corners were found then use the whole map as a submap
    else:
      self.rectilinear_occ_grid[0, 0] = Direction.RIGHT.value
      entire_map = self.makeRectangle(0, 0, self.possible_rectangles)
      self.submaps.append(entire_map)

    # Finally prune out submaps that are smaller than 50 cells or where the minimum size is smaller than our sweeper
    final_submaps = []
    for s in self.submaps:

      # Skip if less than 50 cells
      if len(s.range()) < 50:
        continue

      # Skip if smallest edge of submap is smaller than our block
      if s.is_rectangle and min(block_size_x, block_size_y) > min(s.size_x, s.size_y):
        continue

      final_submaps.append(s)

    self.submaps = final_submaps

    return self.submaps
