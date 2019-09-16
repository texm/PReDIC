import muDIC as dic
import os
import numpy as np
import pandas
import scipy
from muDIC import vlab
import imageio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
def analyse_displacements():
    cwd = os.getcwd()
    path = cwd + r"/example_images"
    image_stack = dic.image_stack_from_folder(path,file_type=".bmp")
    mesher = dic.Mesher()
    mesh = mesher.mesh(image_stack, Xc1=15.0, Xc2=485.0, Yc1=15.0, Yc2=485.0, n_elx=485, n_ely=485, GUI=False)
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
