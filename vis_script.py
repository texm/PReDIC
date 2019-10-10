import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

##------------------------------------~~-------------------------------------##
# To Run:

# python vis_script.py output_file size
# e.g. python vis_script.py  DEF50021.csv 500

# Will create a new folder with the name of the file you are running and replace
# the x, y and residual images inside


def import_output(filename, n):

    deformation = pd.read_csv(filename)
    deformation_np = np.array(deformation)

    # Assign reference and deformed image
    x = deformation_np[:,0:n-1]
    y = deformation_np[:,n:(2*n)-1]

    return x,y

def output_folder(output_name):
    new_dir = 'vis_'+output_name
    if (os.path.exists(new_dir)) == False:
        os.mkdir(new_dir)
    s='/'
    cwd = os.getcwd()
    dir = {cwd, new_dir}
    path=s.join(dir)
    print(path)
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
    output_name = sys.argv[1]
    x, y, = import_output(output_name, int(sys.argv[2]))

    output_folder(output_name[0:-4])

    residual = (x**2 + y**2 )**0.5
    vis_plotter(x, 'X','x_vis.png')
    vis_plotter(y, 'Y','y_vis.png')
    vis_plotter(residual,'Residual' ,'total_vis.png')


main()
