#import C_First_Order
from PIL import Image
import numpy as np
import math
from scipy.interpolate import BPoly
from scipy.interpolate import NdPPoly
import matplotlib.pyplot as plt
global subset_size
global ref_image
global Xp
global Yp
global def_interp
global def_interp_x
global def_interp_y

def DIC_NR_images(ref_img=None,def_img=None,subsetSize=None,ini_guess=None,*args,**kwargs):
    
    # Make sure that the subset size specified is valid (not odd at this point)
    if (subsetSize % 2 == 0):
        raise ValueError("Subset size must be odd")

    # Prepare for trouble (load images) (default directory is current working directory) https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    ref_image = np.array(Image.open(ref_img).convert('LA')) # numpy.array
    def_image = np.array(Image.open(def_img).convert('LA')) # numpy.array

    # Make it double
    ref_image = ref_image.astype('d') # convert to double
    def_image = def_image.astype('d') # convert to double
    print(def_image.shape)
    # Obtain the size of the reference image
    X_size, Y_size, _tmp= ref_image.shape

    # Initialize variables
    subset_size = subsetSize
    spline_order = 6

    # Termination condition for newton-raphson iteration
    Max_num_iter = 40 # maximum number of iterations
    TOL = [0,0]
    TOL[0] = 10**(-8) # change in correlation coeffiecient
    TOL[1] = 10**(-8)/2 # change in sum of all gradients.

    '''
    condition to check that point of interest is not close to edge. Point
    must away from edge greater than half of subset adding 15 to it to have
    range of initial guess accuracy.
    '''
    Xmin = round((subset_size/2) +15) # Might need to make it 14 cuz of arrays starting at 0
    Ymin = Xmin

    Xmax = round(X_size-((subset_size/2) +15))
    Ymax = round(Y_size-((subset_size/2) +15))
    Xp = Xmin
    Yp = Ymin
    print(type(Xp))
    if ( (Xp < Xmin) or (Yp < Ymin) or (Xp > Xmax) or  (Yp > Ymax) ):
        raise ValueError('Process terminated!!! First point of centre of subset is on the edge of the image. ');
    
    #_____________Automatic Initial Guess_____________

    # Automatic Initial Guess
    q_0 = np.zeros_like([], shape=(6))
    q_0[0:2] = ini_guess
    range_ = 15 # Minus 1 for array starting at zero?
    u_check = np.arange((round(q_0[0]) - range_), (round(q_0[1]) + range_)+1, dtype=int)
    v_check = np.arange((round(q_0[0]) - range_), (round(q_0[1]) + range_)+1, dtype=int)

    # Define the intensities of the first reference subset
    subref = ref_image[Yp-math.floor(subset_size/2):(Yp+math.floor(subset_size/2))+1, Xp-math.floor(subset_size/2):Xp+math.floor(subset_size/2)+1,0]
    print(subref)
    
    # Preallocate some matrix space
    sum_diff_sq = np.zeros((u_check.size, v_check.size))
    # Check every value of u and v and see where the best match occurs
    for iter1 in range(u_check.size):
        for iter2 in range(v_check.size):
            subdef = def_image[(Yp-math.floor(subset_size/2)+v_check[iter2]):(Yp+math.floor(subset_size/2)+v_check[iter2])+1, (Xp-math.floor(subset_size/2)+u_check[iter1]):(Xp+math.floor(subset_size/2)+u_check[iter1])+1,0]

            sum_diff_sq[iter2,iter1] = sum(sum(np.square(subref-subdef)))
    print(subdef)
    OFFSET1 = np.argmin(np.min(sum_diff_sq, axis=1)) # These offsets are +1 in MATLAB
    OFFSET2 = np.argmin(np.min(sum_diff_sq, axis=0))
    print(OFFSET1)
    print(OFFSET2)
    q_0[0] = u_check[OFFSET1]
    q_0[1] = u_check[OFFSET2]
    del u_check
    del v_check
    del iter1 
    del iter2 
    del subref 
    del subdef
    del sum_diff_sq
    del OFFSET1 
    del OFFSET2


    # Preallocate the matrix that holds the deformation parameter results
    DEFORMATION_PARAMETERS = np.zeros_like([], shape=(Y_size,X_size,12))

    # Set the initial guess to be the "last iteration's" solution.
    q_k = q_0[1:6]

    #_______________COMPUTATIONS________________

    # Start the timer: Track the time it takes to perform the heaviest computations
    #tic????

    #__________FIT SPLINE ONTO DEFORMED SUBSET________________________
    # Obtain the size of the reference image
    Y_size, X_size,tmp = ref_image.shape
    
    # Define the deformed image's coordinates
    X_defcoord = np.arange(0,X_size, dtype=int) # Maybe zero?
    Y_defcoord = np.arange(0,Y_size, dtype=int)

    #print(np.array([X_defcoord, Y_defcoord]).shape)
    # Fit the interpolating spline: g(x,y) # Which functions to use?
    #def_interp = spapi( [spline_order, spline_order], [Y_defcoord, X_defcoord], def_image[Y_defcoord,X_defcoord] )



    #test = interpolate.splrep()

    spl = NdPPoly(def_image[:,:,0], (Y_defcoord, X_defcoord) )
    exit()
    def_interp = BPoly.from_derivatives(def_image[:,:,0],  [[X_defcoord], [Y_defcoord]], orders=[spline_order, spline_order])
    #print(def_interp)
    # Find the partial derivitives of the spline: dg/dx and dg/dy
    #def_interp_x = fnder(def_interp, [0,1])
    #def_interp_y = fnder(def_interp, [1,0])

    # Convert all the splines from B-form into ppform to make it
    # computationally cheaper to evaluate. Also find partial derivatives of
    # spline w.r.t x and y
    #def_interp = fn2fm(def_interp, 'pp')
    #def_interp_x = fnder(def_interp, [0,1])
    #def_interp_y = fnder(def_interp, [1,0])
    #_________________________________________________________________________ 
    #t_interp = toc;    # Save the amount of time it took to interpolate
    return

    



DIC_NR_images("ref50.bmp", "def50.bmp", 7, [0, 0])