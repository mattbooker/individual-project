from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import PrimitiveShape
import numpy as np
import time
import random

random.seed()

def generate_random_map(total_objects, only_rectilinear=True):

	base = Shape('floor')
	base.set_collidable(False)
	bounding_box = base.get_bounding_box()
	size_x = bounding_box[1] - bounding_box[0]
	size_y = bounding_box[3] - bounding_box[2]

	objects = []

	for i in range(total_objects):
		
		while True:
			#TODO: Set orientation if only_rectilinear is false

			random_x_size = random.uniform(0, 0.7*size_x)
			random_y_size = random.uniform(0, 0.7*size_y)

			random_x_pos = random.uniform(random_x_size/2, size_x - random_x_size/2) - 0.5
			random_y_pos = random.uniform(random_y_size/2, size_y - random_y_size/2) - 0.5
			new_shape = Shape.create(type=PrimitiveShape.CUBOID, size=[random_x_size,random_y_size,.05], position=[random_x_pos,random_y_pos,0.025], static=True, renderable=True)
			new_shape.set_collidable(True)

			if new_shape.check_collision() == False:
				objects.append(new_shape)
				break
			else:
				new_shape.remove()

	return objects

# # Get Scene
# SCENE_FILE = join(dirname(dirname(abspath(__file__))), 'scenes/scene_cpp.ttt')

# # Set numpy printing options
# np.set_printoptions(threshold=100*100, formatter={'all':lambda x: str(x) + ','})

# # Start Simulation
# pr = PyRep()
# pr.launch(SCENE_FILE, headless=False)
# pr.start()
# pr.step()

# generate_random_map(3)
# pr.step()
# time.sleep(2)

# # End Simulation
# pr.stop()
# pr.shutdown()

