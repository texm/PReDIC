import muDIC as dic
import os
import numpy as np
import pandas
import scipy
from muDIC import vlab
import imageio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def deform_image(image):
    pass
    #for i in len(image.shape[0]):
    #    print(i)
        
    #return image

def main():
    speckle = vlab.dots_speckle((250,250),n_dots=5000,dot_radius_max=10,dot_radius_min=5,blur_sigma=1,allow_overlap=False)
    F = np.array([[1.5,.0], [0., 1.1]], dtype=np.float64)
    #vlab.imageDeformer_from_uFunc
    image_deformer = vlab.imageDeformer_from_defGrad(F)
    speckle_image = image_deformer(speckle)[1]
    #deform_image(speckle_image)

    #print(speckle.min())
    #speckle *= 255
    #speckle = speckle.astype(np.uint8)
    
    speckle_image *= 255
    speckle_image = speckle_image.astype(np.uint8)
    imageio.imwrite('ref250.bmp', speckle)
    imageio.imwrite('def250,x=2.1,y=1.1.bmp', speckle_image)
    #plt.imshow(speckle, 'gray')
    #plt.imshow(speckle_image, 'gray')
    #plt.show()
    #dic.DICOutput.Ic_stack.
main()
