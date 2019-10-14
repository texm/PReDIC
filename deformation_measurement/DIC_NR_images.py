from deformation_measurement.C_First_Order import C_First_Order
#import C_First_Order
import deformation_measurement.Globs as Globs
#import Globs

from math import floor
import numpy as np
from PIL import Image
from scipy.interpolate import splrep, PPoly, RectBivariateSpline, BSpline, BPoly, bisplev, bisplrep, splder

def initial_guess(ref_img, def_img, ini_guess, subset_size, Xp, Yp):

    # Automatic Initial Guess
    #q_0 = np.zeros_like([], shape=6)
    q_0 = np.zeros(6)
    q_0[0:2] = ini_guess

    range_ = 15 # Minus 1 for array starting at zero?
    u_check = np.arange((round(q_0[0]) - range_), (round(q_0[1]) + range_) + 1, dtype=int)
    v_check = np.arange((round(q_0[0]) - range_), (round(q_0[1]) + range_) + 1, dtype=int)

    # Define the intensities of the first reference subset
    subref = ref_img[Yp-floor(subset_size/2):(Yp+floor(subset_size/2))+1, Xp-floor(subset_size/2):Xp+floor(subset_size/2)+1,0]
    
    # Preallocate some matrix space
    sum_diff_sq = np.zeros((u_check.size, v_check.size))
    # Check every value of u and v and see where the best match occurs
    for iter1 in range(u_check.size):
        for iter2 in range(v_check.size):

            #Define intensities for deformed subset
            subdef = def_img[(Yp-floor(subset_size/2)+v_check[iter2]):(Yp+floor(subset_size/2)+v_check[iter2])+1, (Xp-floor(subset_size/2)+u_check[iter1]):(Xp+floor(subset_size/2)+u_check[iter1])+1,0]

            #extra sum here?
            sum_diff_sq[iter2,iter1] = np.sum(np.sum(np.square(subref-subdef)))

    OFFSET1 = np.argmin(np.min(sum_diff_sq, axis=1)) # These offsets are +1 in MATLAB
    OFFSET2 = np.argmin(np.min(sum_diff_sq, axis=0))

    q_0[0] = u_check[OFFSET2]
    q_0[1] = v_check[OFFSET1]

    q_k = q_0[0:6]

    return q_k


def fit_spline(ref_img, def_img, spline_order):

    # Obtain the size of the reference image
    Y_size, X_size,tmp = ref_img.shape

    # Define the deformed image's coordinates
    X_defcoord = np.arange(0, X_size, dtype=int) # Maybe zero?
    Y_defcoord = np.arange(0, Y_size, dtype=int)

    #Fit spline
    def_interp = RectBivariateSpline(X_defcoord, Y_defcoord, def_img[:,:,0], kx=spline_order-1, ky=spline_order-1)
    #why subtract 1 from spline order?

    #Evaluate derivatives at coordinates
    def_interp_x = def_interp(X_defcoord, Y_defcoord, 0, 1)
    def_interp_y = def_interp(X_defcoord, Y_defcoord, 1, 0)

    return def_interp, def_interp_x, def_interp_y


def DIC_NR_images(ref_img=None,def_img=None,subsetSize=None,ini_guess=None,*args,**kwargs):
    
    Globs.subset_size = subsetSize

    # Make sure that the subset size specified is valid (not odd at this point)
    if (Globs.subset_size % 2 == 0):
        raise ValueError("Subset size must be odd")

    # Prepare for trouble (load images) (default directory is current working directory) https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    Globs.ref_image = np.array(Image.open(ref_img).convert('LA')) # numpy.array
    Globs.def_image = np.array(Image.open(def_img).convert('LA')) # numpy.array

    # Make it double
    Globs.ref_image = Globs.ref_image.astype('d') # convert to double
    Globs.def_image = Globs.def_image.astype('d') # convert to double

    # Obtain the size of the reference image
    X_size, Y_size, _tmp= Globs.ref_image.shape

    # Initialize variables
    subset_size = Globs.subset_size
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
    Xmin = round((subset_size/2) + 15) # Might need to make it 14 cuz of arrays starting at 0
    Ymin = Xmin

    Xmax = round(X_size-((subset_size/2) + 15))
    Ymax = round(Y_size-((subset_size/2) + 15))
    Globs.Xp = Xmin
    Globs.Yp = Ymin

    if ( (Globs.Xp < Xmin) or (Globs.Yp < Ymin) or (Globs.Xp > Xmax) or  (Globs.Yp > Ymax) ):
        raise ValueError('Process terminated!!! First point of centre of subset is on the edge of the image. ')
    
    #_____________Automatic Initial Guess_____________

    #Calculate quick guess for u&v through sum of differences squared?
    # Set the initial guess to be the "last iteration's" solution.
    q_k = initial_guess(Globs.ref_image, Globs.def_image, ini_guess, subset_size, Globs.Xp, Globs.Yp)

    # Preallocate the matrix that holds the deformation parameter results
    DEFORMATION_PARAMETERS = np.zeros_like([], shape=(Y_size,X_size,12))


    #_______________COMPUTATIONS________________

    # Start the timer: Track the time it takes to perform the heaviest computations
    #tic????

    #__________FIT SPLINE ONTO DEFORMED SUBSET________________________

    Globs.def_interp, Globs.def_interp_x, Globs.def_interp_y = fit_spline(Globs.ref_image, Globs.def_image, spline_order)

    #_________________________________________________________________________ 
    #t_interp = toc;    # Save the amount of time it took to interpolate


    # MAIN CORRELATION LOOP -- CORRELATE THE POINTS REQUESTED

    # for i=1:length(pts(:,1))
    for yy in range(Ymin, Ymax + 1):
        if yy > Ymin:
            q_k[0:6] = DEFORMATION_PARAMETERS[yy - 1, Xmin, 0:6]

        for xx in range(Xmin, Xmax + 1):
            #Points for correlation and initializaing the q matrix
            Globs.Xp = xx + 1
            Globs.Yp = yy + 1
            #t_tmp = toc

            # __________OPTIMIZATION ROUTINE: FIND BEST FIT____________________________
            # if (itr_skip == 0)
            # Initialize some values
            n = 0
            C_last, GRAD_last, HESS = C_First_Order(q_k) # q_k was the result from last point or the user's guess
            optim_completed = False

            if np.isnan(abs(np.mean(np.mean(HESS)))):
                print(yy)
                print(xx)
                optim_completed = True
            while not optim_completed:
                # Compute the next guess and update the values
                delta_q = np.linalg.lstsq(HESS,(-GRAD_last), rcond=None) # Find the difference between q_k+1 and q_k
                q_k = q_k + delta_q[0]                             #q_k+1 = q_k + delta_q[0]
                C, GRAD, HESS = C_First_Order(q_k) # Compute new values
                
                # Add one to the iteration counter
                n = n + 1 # Keep track of the number of iterations

                # Check to see if the values have converged according to the stopping criteria
                if n > Max_num_iter or (abs(C-C_last) < TOL[0] and all(abs(delta_q[0]) < TOL[1])): #needs to be tested...
                    optim_completed = True
                
                C_last = C #Save the C value for comparison in the next iteration
                GRAD_last = GRAD # Save the GRAD value for comparison in the next iteration
            #_________________________________________________________________________
            #t_optim = toc - t_tmp

            #_______STORE RESULTS AND PREPARE INDICES OF NEXT SUBSET__________________
            # Store the current displacements
            DEFORMATION_PARAMETERS[yy,xx,0] = q_k[0] # displacement x
            DEFORMATION_PARAMETERS[yy,xx,1] = q_k[1] # displacement y
            DEFORMATION_PARAMETERS[yy,xx,2] = q_k[2] 
            DEFORMATION_PARAMETERS[yy,xx,3] = q_k[3] 
            DEFORMATION_PARAMETERS[yy,xx,4] = q_k[4] 
            DEFORMATION_PARAMETERS[yy,xx,5] = q_k[5] 
            DEFORMATION_PARAMETERS[yy,xx,6] = 1-C # correlation co-efficient final value
            # store points which are correlated in reference image i.e. center of subset
            DEFORMATION_PARAMETERS[yy,xx,7] = Globs.Xp 
            DEFORMATION_PARAMETERS[yy,xx,8] = Globs.Yp

            DEFORMATION_PARAMETERS[yy,xx,9] = n # number of iterations
            DEFORMATION_PARAMETERS[yy,xx,10] = 0 #t_tmp # time of spline process
            DEFORMATION_PARAMETERS[yy,xx,11] = 0 #t_optim # time of optimization process

    '''
        print(yy)
        print(xx)


    filename = f"DEFORMATION_PARAMETERS({ref_img}, {def_img}, {Globs.subset_size})".replace('/', '')
    xxx,yyy,zzz = DEFORMATION_PARAMETERS.shape
    sav = np.swapaxes(DEFORMATION_PARAMETERS,2,1).reshape((xxx,yyy*zzz), order='A')
    savetxt_compact(filename, sav)
    savetxt_compact_matlab(filename, sav)
    '''

    return []

def savetxt_compact(fname, x, fmt="%.6g", delimiter=','):
    with open(f"compact_{fname}.csv", 'w+') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

def savetxt_compact_matlab(fname, x, fmt="%.6g", delimiter=','):
    with open(f"matlab_{fname}.csv", 'w+') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

#DIC_NR_images("ref50.bmp", "def50.bmp", 7, [0, 0])
