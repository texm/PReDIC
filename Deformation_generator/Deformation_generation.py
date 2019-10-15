import sys
import numpy as np
from scipy.interpolate import griddata
from PIL import Image
from math import *

def test():
	for arg in sys.argv:
		print(arg)

	Deformation_generation(sys.argv[1])

def Deformation_generation(ref_img):
	#open image
	im = Image.open("Reference_images/" + str(ref_img))
	im.resize((50,50))
	width, height = im.size

	#convert to array
	im_array = np.array(im)

	#we transform these to get coordinates of where the pixel data will be in the second image
	#grid_y is y coord, grid_x is x coord
	#note: these are indexed from the upper left corner of the image
	grid_y,grid_x = np.mgrid[0:height, 0:width]

	#transform y & x coord according to some function
	trans_y = grid_y*2
	trans_x = grid_x*2

	#reshape to combine into (x,y) pairs
	flat_y = np.reshape(trans_y, width*height)
	flat_x = np.reshape(trans_x, width*height)

	#x & y coordinates of data points after they have been transformed
	coords = np.vstack((flat_y,flat_x)).T

	#associated pixel data point for each coordinate
	values = np.reshape(im_array, width*height)

	#interpolate image based on transformed coords and pixel data
	#this places pixel data in 'values' at the coordinate in 'coords', the values in grid_x and grid_y are then calculated by interpolation.
	grid = griddata(coords,values,(grid_x, grid_y),method = 'linear')

	#calculate the displacements
	y_disp = trans_y - grid_y
	x_disp = trans_x - grid_x

	#save displacements to file
	np.savetxt("x_disp.csv", x_disp, delimiter=",")
	np.savetxt("y_disp.csv", y_disp, delimiter=",")

	#display deformed image
	#should check saving format ***
	out = Image.fromarray(grid).convert('L')
	out.save('Deformed_images/def250.bmp')
	out.show()



def main():
    print("Hello World!")
    test()

if __name__ == "__main__":
    main()

















