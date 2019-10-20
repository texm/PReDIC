import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import imageio
from math import sin,cos
from PIL import Image


'''
Jiggle Maker function for CITS33200 deformation images
    - Produces a reference and deformation images depending on the
      imput deformations
    - Produce a "ground truth" to later compare images with

To Run: python3 jiggle_maker.py
'''


##------------------------------------~~-------------------------------------##
# Set up parameters:
#--------------------

# Min & max - sets the xy limits of the image
min = 0
max = 100


size = 2000     # The number of speckles #usual for 100x100: 1500
inch_size = 1    # The size of the image
dpi = 500        # Dots per inch - will set the pixel width of the image


dot_rad = 0.5  # How big you want the circles to be plotted

ref_name = 'ref500'
def_name = 'def500'
gif_name = 'test500.gif'
gt_name  = 'gt500'

##------------------------------------~~-------------------------------------##
# Deformation functions:
#-----------------------


def deformation_x(x):
    return 1.0101010101*x




def deformation_y(y):
    return 1.0101010101*y

# note: 0.1 = 1 pixel in 500x500 version
##------------------------------------~~-------------------------------------##


# To make the images the same each time (turn this off to generate more images)
np.random.seed(seed=19)

# Use a uniform random distribution
initial_x = np.random.uniform(min, max, size)
initial_y = np.random.uniform(min, max, size)

dpi_inch = np.floor(dpi/inch_size) # Sets the number of pixels you need

def to_bmp(name):
    # Open the file, convert to grey scale
    img = Image.open(name+'.png').convert('L')
    file_out = name + ".bmp"
    img.save(file_out, cmap='gray')

# Plot the image with the required format
def image_plot(x,y,name):
    fig, ax = plt.subplots()
    fig.set_size_inches(inch_size,inch_size)

    # Need to use patches class to get sub-pixel accuracy
    initial_patches=[]

    for i in range(size):
        initial_patches.append(plt.Circle((x[i],y[i]),radius=dot_rad, fc='k', ec='k'))
    ax.add_collection(PatchCollection(initial_patches, edgecolors='k', facecolors='k'))

    # For removing the axis and plotting to the very edge of the image
    ax.set_xlim([min,max])
    ax.set_ylim([min,max])
    ax.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,hspace = 0, wspace = 0)

    # Save image in the required format
    plt.savefig( name , bbox_inches=0, pad_inches = 0, dpi=dpi_inch)
    plt.close()
    to_bmp(name)


# Makes a gif for visualisation purposes
def gif_generator():
    images = []
    images.append(imageio.imread(ref_name+'.png'))
    images.append(imageio.imread(def_name+'.png'))

    imageio.mimsave(gif_name, images)


# Return the number of pixels that have not been deformed as we expected
# Because of 'subpixel accuracy - only works for translations'
def check_translations(ref_name, def_name):

    # Load in given images and read their pixels
    ref_img = Image.open(ref_name+'.bmp')
    ref_pixel = ref_img.load()
    def_img = Image.open(def_name+'.bmp')
    def_pixel = def_img.load()

    count = 0
    for test_x in range(min, max):

        for test_y in range(min, max):
            test_x_def = deformation_x(test_x)
            test_y_def = deformation_y(test_y)

            # Points that have been translated outside the image
            if test_x_def >= max or test_y_def >= max or test_x_def <= min or test_y_def <= min:
                continue

            # Check for how many pixels we incorrectly guessed the deformation
            if def_pixel[test_x_def,test_y_def] != ref_pixel[test_x,test_y]:
                count +=1

    return count

def vis_plotter(array,arr_name,im_name):
    fig, ax = plt.subplots()
    imgplot = plt.imshow(array)
    imgplot.set_cmap('YlGnBu')
    ax.set_title(arr_name+" Deformations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar()
    plt.savefig(im_name)
    plt.close()

def ground_truth(gt_name):
    def_x_arr = []
    def_y_arr = []

    for init_x in range(min,max):
        temp_x_arr = []
        temp_y_arr = []
        for init_y in range(min,max):
            fin_x = deformation_x(init_x)
            fin_y = deformation_y(init_y)

            delta_x = fin_x - init_x
            delta_y = fin_y - init_y

            temp_x_arr.append(delta_x)
            temp_y_arr.append(delta_y)

        def_x_arr.append(temp_x_arr)
        def_y_arr.append(temp_y_arr)

    # Convert to numpy arrays
    def_x_arr = np.rot90(np.asarray(def_x_arr))
    def_y_arr = np.rot90(np.asarray(def_y_arr))

    residual_diff = (def_x_arr**2 + def_y_arr**2)**0.5


    vis_plotter(def_x_arr, 'X ground truth', gt_name+'x.png')
    vis_plotter(def_y_arr, 'Y ground truth', gt_name+'y.png')
    vis_plotter(residual_diff, 'Residual ground truth', gt_name+'r.png')

    final = np.concatenate((def_x_arr,def_y_arr) ,axis=1)

    np.savetxt(gt_name+".csv", final, delimiter=",")

##------------------------------------~~-------------------------------------##
# Calling functions:

image_plot(initial_x, initial_y, ref_name)
image_plot(deformation_x(initial_x), deformation_y(initial_y), def_name)
gif_generator()
ground_truth(gt_name)

# Testing for linear translations - subpixel accuracy means stretches wont work
# If we have more than 1 incorrectly prediccted point - notify

#if check_translations(ref_name, def_name) > 0:
#    print('WARNING: not translating as expected')
