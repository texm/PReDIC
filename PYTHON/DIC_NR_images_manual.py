#import C_First_Order
import matplotlib.image
import numpy
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

    # Prepare for trouble (load images) (default directory is current working directory)
    ref_image = matplotlib.image.imread(ref_img) # numpy.array
    def_image = matplotlib.image.imread(def_img) # numpy.array

    # Make it double
    ref_image = ref_image.astype('d') # convert to double
    def_image = def_image.astype('d') # convert to double

    # Obtain the size of the reference image
    X_size, Y_size = ref_image.shape

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

    if ( (Xp < Xmin) or (Yp < Ymin) or (Xp > Xmax) or  (Yp > Ymax) ):
        raise ValueError('Process terminated!!! First point of centre of subset is on the edge of the image. ');
    
    #_____________Automatic Initial Guess_____________

    # Automatic Initial Guess
    q_0 = numpy.zeros_like([], shape=(6,1))
    q_0 = numpy.zeros_like([], shape=(6)).T
    print(q_0)
    q_0[0:2,0] = ini_guess
    range_ = 15 # Minus 1 for array starting at zero?
    print(q_0[0][0])
    u_check = numpy.arange((round(q_0[0]) - range_), (round(q_0[1]) + range_))
    print(u_check)
    return

DIC_NR_images("ref500.bmp", "def500.bmp", 7, [4, 2])