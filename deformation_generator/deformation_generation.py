import sys
import numpy as np
from PIL import Image
from math import *
import math
import os
import cairo

#Requires pycairo & pkg-config
#Refer here: https://pycairo.readthedocs.io/en/latest/getting_started.html

#These functions allow for the generation of speckle images.
#Images are generated via the use of a vector graphics library pycairo.
#This allows for image transformations without the need for interpolation or resampling error consideration.
#Precision necessary when measuring accuracy of DIC algorithm to sub-pixel level. 
#The x and y displacements can then be calculated precisely according to the transformation.

#Matrix transforms used for deformations
#[a1, c3, e5,
# b2, d4, f6,
# 0, 0, 1] these numbers relate to the .Matrix(1,2,3,4,5,6) variables

# x' = a1*x + c3*y + e5
# y' = d4*y + b2*x + f6

# (a1&d4):provide scale
# (c3&b2):provide shear
# (ef+f6):provide translation

#Function will generate reference image, deformed image and provide x&y translation arrays
def generate_images(image_size,seed,a1,b2,c3,d4,e5,f6):
	gen_ref(image_size, seed)
	gen_def(image_size, seed, a1,b2,c3,d4,e5,f6)
	xd,yd = calc_translations(image_size,a1,b2,c3,d4,e5,f6)
	print(xd,yd)

#Will draw speckles using uniform random distribution, change seed for different speckle pattern
def draw_speckles(context, seed):

	#Create white background
	context.set_source_rgb(1, 1, 1)
	context.rectangle(0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1)
	context.fill()

	#Change colour to black
	context.set_source_rgb(0,0,0)
	context.move_to(0, 0)

	size = 3000 # The number of speckles

	# To make the images the same each time (matching ref & def)
	np.random.seed(seed= seed)
	min = 0
	max = 1

	# Use a uniform random distribution
	initial_x = np.random.uniform(min, max, size)
	initial_y = np.random.uniform(min, max, size)

	for i in range(size):
		#circle (xc, yc, radius, start_angle, end_angle)
		context.arc(initial_x[i], initial_y[i], 0.01, 0, 2*math.pi)
		context.fill()

#Calculates x and y displacements between reference image and deformed image, according to transformation matrix
def calc_translations(image_size,a1,b2,c3,d4,e5,f6):

	#create transformation matrix
	trans_matrix = [[a1,c3],[b2,d4]]

	#original x and y coordinates
	orig_y, orig_x = np.mgrid[1:image_size + 1,1:image_size+1]

	#x,y pairs to be transformed
	xy_points = np.mgrid[1:image_size + 1,1:image_size + 1].reshape((2,image_size*image_size))
	xy_points[[1,0]] = xy_points[[0,1]]

	#transformed x,y pairs
	new_points = np.dot(trans_matrix,xy_points)
	x, y = new_points.reshape((2,image_size,image_size))

	#add translation element of matrix
	x = np.add(x, e5).reshape((image_size,image_size))
	y = np.add(y, f6).reshape((image_size,image_size))

	#calculate the x and y displacements
	xd = (x - orig_x)
	yd = (y - orig_y)

	return xd,yd	

#Generate reference images
def gen_ref(image_size, seed):
	WIDTH, HEIGHT = image_size, image_size

	surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
	context = cairo.Context(surface)

	context.scale(WIDTH, HEIGHT)  # Normalizing the canvas

	draw_speckles(context, seed)

	context.close_path()

	img_name = "ref_" + str(image_size) + "_" + str(seed) + ".bmp"

	write_image(surface, image_size, img_name)

#Generate deformed images
def gen_def(image_size, seed, a1,b2,c3,d4,e5,f6):
	WIDTH, HEIGHT = image_size, image_size

	format = cairo.FORMAT_ARGB32

	surface = cairo.ImageSurface(format, WIDTH, HEIGHT)
	context = cairo.Context(surface)

	# Normalizing the canvas
	context.scale(WIDTH, HEIGHT)  

	#Matrix transform
	#[a1, c3, e5,
	# b2, d4, f6,
	# 0, 0, 1] these numbers relate to the .Matrix(1,2,3,4,5,6) variables

	# x' = a1*x + c3*y + e5
	# y' = d4*y + b2*x + f6

	# (a1&d4):provide scale
	# (c3&b2):provide shear
	# (ef+f6):provide translation

	mtx = cairo.Matrix(a1,b2,c3,d4,e5,f6)
	context.transform(mtx)

	draw_speckles(context, seed)

	context.close_path()

	img_name = "def_" + str(image_size) + "_" + str(seed) + ".bmp"
	np_matrix = np.array([a1, c3, e5],[b2,d4,f6],[0,0,1])
	write_image(surface, image_size, img_name, np_matrix, seed)

#Writes image to /img_gen directory in format as specified by filename (currently works for .bmp)
def write_image(surface, image_size, file_name, np_matrix, seed):
	save_dir = os.path.dirname(os.path.realpath(__file__)) + "/img_gen"

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	buf = surface.get_data()
	data = np.ndarray(shape=(image_size, image_size), dtype=np.uint32,buffer=buf)

	out = Image.fromarray(data, 'RGBA')
	out.save(save_dir +"/"+file_name)
	with open(save_dir +"/"+file_name+"-metadata.txt", "a") as file:
		file.write(image_size + "\n")
		file.write(seed + "\n")
		np.savetxt(file, np_matrix)
	
def main():
    generate_images(50,19,1.1, 0.0, 0.0, 1, 0.0, 0.0)


    #generate_images(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8])

    #for arg in sys.argv:
	#	print(arg)

	#Deformation_generation(sys.argv[1])

if __name__ == "__main__":
    main()

















