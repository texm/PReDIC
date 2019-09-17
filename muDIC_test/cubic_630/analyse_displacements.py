import muDIC as dic
import os
import numpy as np
import pandas
import scipy
from muDIC import vlab
import imageio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def find_file_names(path, type=".png"):
    """
    Finds all files with the given extension in the folder path.

     Parameters
     ----------
     path : str
         The path to the folder containing the files of interest
     type : str
         The file postfix such as ".png", ".bmp" etc.

     Returns
     -------
    List of filenames

     """
    return sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith(type)])

def analyse_displacements():
    cwd = os.getcwd()
    path = cwd
    image_stack = dic.image_stack_from_folder(path,file_type=".bmp")
    '''
    height, width, channels = 0,0,0
    image = find_file_names(path, type=".bmp")[0]
    height, width, channels = scipy.ndimage.imread(image).shape
    '''

    mesher = dic.Mesher()
    #mesh = mesher.mesh(image_stack, Xc1=15.0, Xc2=485.0, Yc1=15.0, Yc2=485.0, n_elx=32, n_ely=32, GUI=False)
    mesh = mesher.mesh(image_stack, Xc1=30.0, Xc2=600.0, Yc1=30.0, Yc2=600.0, n_elx=400, n_ely=400, GUI=False)
    #mesh = mesher.mesh(image_stack)
    #mesh = mesher.mesh(image_stack, Xc1=25.0, Xc2=35.0, Yc1=25.0, Yc2=35.0, n_elx=1, n_ely=1, GUI=False)
    inputs = dic.DICInput(mesh,image_stack)
    dic_job = dic.DICAnalysis(inputs)
    results = dic_job.run()
    fields = dic.Fields(results)
    displ = fields.disp()
    #true_strain = fields.true_strain()
    #print(displ)
    #numpy.savetxt("foo.csv", displ, delimiter=",")
    #print(displ.shape)
    viz = dic.Visualizer(fields,images=image_stack)
    viz.show(field="u", component = (1,1), frame = 1)

analyse_displacements()
