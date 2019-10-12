import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os


def output_folder(folder_name):
    if (os.path.exists(folder_name)) == False:
        os.mkdir(folder_name)
    cwd = os.getcwd()
    path= cwd + "/" + folder_name+'/'
    os.chdir(path)


def vis_plotter(array,arr_name,im_name):
    ig, ax = plt.subplots()
    imgplot = plt.imshow(array)
    imgplot.set_cmap('YlGnBu')
    ax.set_title(arr_name+" Deformations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar()
    plt.savefig(im_name)
    plt.close()
