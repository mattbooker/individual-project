import math
import random
import time

import matplotlib.pyplot as plt

from pyrep import PyRep
from pyrep.objects.shape import Shape


class RRT:

  '''
  Define a node class that represents a node in our RRT
  '''
  class Node:
    def __init__(self, x, y, theta):
      self.x = x
      self.y = y
      self.theta = theta
      self.parent = None
      
      self.path = []

    def distance_to(self, node):
      return math.sqrt((self.x - node.x)**2 + (self.y - node.y)**2)


  rrt_tree = []

  def __init__(self, pr, shape, start_pose, target_pose, limits_x, limits_y):
    random.seed()

    self.pr = pr

    self.shape = shape
    self.start_pose = start_pose
    self.target_pose = target_pose
    self.rrt_tree.append(self.Node(start_pose[0], start_pose[1], start_pose[2]))

    self.min_x, self.max_x = limits_x
    self.min_y, self.max_y = limits_y

    self.max_iterations = 2000
    self.edge_length = 0.05
    self.path_resolution = 0.01


  # TODO: dont forget to move the shape back to start pose
  def solve(self):
    goal = self.Node(self.target_pose[0], self.target_pose[1], self.target_pose[2])

    for i in range(self.max_iterations):
      q_rand = self.sample_free()
      q_near = self.getNearestNeighbour(q_rand)

      q_new = self.steer(q_near, q_rand)

      if i % 20 == 0:
        self.draw_graph(q_rand)

      if q_new != None and self.check_collision(q_new.x, q_new.y, q_new.theta) == False:
        self.rrt_tree.append(q_new)
  
        if q_new.distance_to(goal) <= self.edge_length:
          final_node = self.steer(q_new, goal)
          
          if final_node != None and self.check_collision(final_node.x, final_node.y, final_node.theta) == False:
            self.rrt_tree.append(final_node)
            return self.get_path()


    print("ERROR: NO PATH FOUND")
    return None

  def check_collision(self, x, y, theta):
    # Move the shape so that we can test collision
    self.shape.set_position([x, y, self.shape.get_position()[2]])
    self.shape.set_orientation([0, 0, theta])

    return self.shape.check_collision()

  # Goal region biasing
  def sample_free(self):

    i = random.random()
    # 10% of the time put the goal as the random node
    if i >= 0.9:
      return self.Node(self.target_pose[0], self.target_pose[1], self.target_pose[2])

    while True:
      rand_x, rand_y, rand_theta = self.sample()    

      if(self.check_collision(rand_x, rand_y, rand_theta) == False):
        return self.Node(rand_x, rand_y, rand_theta)

  def sample(self):
    rand_x = random.uniform(self.min_x, self.max_x)
    rand_y = random.uniform(self.min_y, self.max_y)
    rand_theta = random.uniform(math.radians(-180), math.radians(180))

    return (rand_x, rand_y, rand_theta) 

  # Adapted from https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
  def steer(self, from_node, to_node):
    
    distance, required_angle = self.calculate_motion(from_node, to_node)

    new_node = self.Node(from_node.x, from_node.y, from_node.theta)
    new_node.path = [(new_node.x, new_node.y, new_node.theta)]

    extend_length = min(distance, self.edge_length)

    n_expand = math.floor(extend_length / self.path_resolution)

    angle_to_rotate_by = self.normalizePi(required_angle - new_node.theta)

    # new_node.theta = required_angle

    if n_expand != 0:
      angle_step = angle_to_rotate_by / n_expand

    for _ in range(n_expand):
        new_node.x += self.path_resolution * math.cos(required_angle)
        new_node.y += self.path_resolution * math.sin(required_angle)
        new_node.theta += angle_step

        if self.check_collision(new_node.x, new_node.y, new_node.theta):
          return None

        new_node.path.append((new_node.x, new_node.y, new_node.theta))

    distance, _ = self.calculate_motion(new_node, to_node)

    if distance <= self.path_resolution:
        new_node.path.append((to_node.x, to_node.y, self.normalizePi(required_angle)))

    new_node.parent = from_node

    return new_node


  # Returns the node in the rrt_tree that has the minimum distance to query_node
  def getNearestNeighbour(self, query_node):
    return min(self.rrt_tree, key=lambda x: x.distance_to(query_node))

  def calculate_motion(self, from_node, to_node):
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    theta = math.atan2(dy, dx)
    return distance, theta

  # Keep any angle we move through between -pi and pi
  def normalizePi(self, val):
    while val < -math.pi:
        val += 2*math.pi

    while val > math.pi:
        val -= 2*math.pi

    return val

  def get_path(self):
    cur_node = self.rrt_tree[-1]
    path = [(cur_node.x, cur_node.y, cur_node.theta)]

    for i in cur_node.path[::-1]:
      path.append(i)

    while cur_node.parent != None:
      cur_node = cur_node.parent
      path.append((cur_node.x, cur_node.y, cur_node.theta))
      
      for i in cur_node.path[::-1]:
        path.append(i)

    return path[::-1]

  def draw_graph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.rrt_tree:
            if node.parent:
                x = [i[0] for i in node.path]
                y = [i[1] for i in node.path]
                plt.plot(x, y, "-g")

            plt.plot(node.x, node.y, "o")


        plt.plot(self.start_pose[0], self.start_pose[1], "xr")
        plt.plot(self.target_pose[0], self.target_pose[1], "xr")
        plt.axis("equal")
        # plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)