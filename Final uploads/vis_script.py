import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os


def import_output(filename, n):

    deformation = pd.read_csv(filename)
    deformation_np = np.array(deformation)

    # Assign reference and deformed image
    x = deformation_np[:,0:n-1]
    y = deformation_np[:,n:(2*n)-1]

    print(deformation_np.shape)

    return x,y

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

def main():

    x, y, = import_output(sys.argv[1],int(sys.argv[2]))

    output_folder(sys.argv[3])

    residual = (x**2 + y**2 )**0.5
    vis_plotter(x, 'X','x_vis.png')
    vis_plotter(y, 'Y','y_vis.png')
    vis_plotter(residual,'Residual' ,'total_vis.png')


main()
