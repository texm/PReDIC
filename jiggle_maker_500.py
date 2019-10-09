import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import imageio
from math import sin,cos

'''
Jiggle Maker function for CITS33200 deformation images
    - Produces a reference and deformation images depending on the
      imput deformations

To Run: python3 jiggle_maker.py
'''


##------------------------------------~~-------------------------------------##
# Set up parameters:
#--------------------

# Min & max - sets the xy limits of the image
# e.g. plotting values from 0 - 50
min = 0
max = 50


size = 10000     # The number of speckles
inch_size = 2    # The size of the image (client image ~ 2 inch)
dpi = 500        # Dots per inch - will set the pixel width of the image


dot_rad = 0.1  # How big you want the circles to be plotted

ref_name = 'ref500.png'
def_name = 'def500.png'
gif_name = 'test500.gif'

##------------------------------------~~-------------------------------------##
# Deformation functions:
#-----------------------

def deformation_x(x):
    #return 1.005*x**1.001 # stretch in the x
    return x+0.001



def deformation_y(y):
    return y

# note: 0.1 = 1 pixel
##------------------------------------~~-------------------------------------##


# To make the images the same each time (turn this off to generate more images)
np.random.seed(seed=19)

# Use a uniform random distribution
initial_x = np.random.uniform(min, max, size)
initial_y = np.random.uniform(min, max, size)

dpi_inch = np.floor(dpi/inch_size) # Sets the number of pixels you need



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


# Makes a gif for visualisation purposes
def gif_generator():
    images = []
    images.append(imageio.imread(ref_name))
    images.append(imageio.imread(def_name))
    imageio.mimsave('test.gif', images)


##------------------------------------~~-------------------------------------##
# Calling functions:
image_plot(initial_x, initial_y, ref_name)
image_plot(deformation_x(initial_x), deformation_y(initial_y), def_name)
gif_generator()
