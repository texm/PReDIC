from speckle_pattern import generate_and_save
from imageio import get_writer
import numpy as np
import os

def generate_images():
	height = 40
	width = 40
	av_diameter = 1
	blur = 0
	dpi = 400
	save_dir = os.path.dirname(os.path.realpath(__file__)) + "/img_gen"
	ref_i = save_dir + "/ref.jpg"
	def_i = save_dir + "/def.jpg"

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	image = generate_and_save(height, width, dpi, av_diameter, ref_i, speckle_blur=blur)

	for x in range(128, 256):
		for y in range(128, 256):
			prev = image[x][y]
			image[x][y] = image[x - 1][y - 1]
			image[x - 1][y - 1] = prev

	with get_writer(def_i, mode='i') as writer:
		writer.append_data(np.uint8(image/np.max(image)*255), meta={"quality": 100})

	return (ref_i, def_i)