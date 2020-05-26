import math
import random
import time
from os.path import abspath, dirname, join

import numpy as np
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor


def generate_random_map(total_objects, rectilinear=True):

	# Set the random seed in the main module

	base = Shape('floor')
	base.set_collidable(False)
	bounding_box = base.get_bounding_box()
	size_x = bounding_box[1] - bounding_box[0]
	size_y = bounding_box[3] - bounding_box[2]

	objects = []

	for i in range(total_objects):
		
		while True:

			random_x_size = random.uniform(0.1*size_x, 0.5*size_x)
			random_y_size = random.uniform(0.1*size_y, 0.5*size_y)



			random_x_pos = random.uniform(random_x_size/2, size_x - random_x_size/2) - 0.5
			random_y_pos = random.uniform(random_y_size/2, size_y - random_y_size/2) - 0.5
			
			yaw = 0 if rectilinear else random.uniform(0, math.pi)

			new_shape = Shape.create(type=PrimitiveShape.CUBOID, 
															 size=[random_x_size,random_y_size,.05], 
															 position=[random_x_pos,random_y_pos,0.025], 
															 orientation=[0,0,yaw],
															 static=True, 
															 renderable=True)

			new_shape.set_collidable(True)

			if new_shape.check_collision() == False:
				objects.append(new_shape)
				break
			else:
				new_shape.remove()

	return objects
