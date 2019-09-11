# Generated with SMOP  0.41


# Please see https://bsplines.org/scientific-programming-with-b-splines/
from smop.libsmop import *
import matplotlib.image
from scipy.interpolate import BPoly
import C_First_Order
# DIC_NR_images.m

    
    # Example Usage: DIC_NR_images("ref500.bmp", "def500.bmp", 7, [0 0])

global subset_size
global ref_image
global Xp
global Yp
global def_interp
global def_interp_x
global def_interp_y    
@function
def DIC_NR_images(ref_img=None,def_img=None,subsetSize=None,ini_guess=None,*args,**kwargs):
    varargin = DIC_NR_images.varargin
    nargin = DIC_NR_images.nargin

    # Variable ref_image takes B&W reference image
# Variable def_image takes B&W deformed image
# Variable subset_size must be provided with positive integer number
# mentioning the size of the subset to be used for correlation.
# Deformations greater than half of subset size cannot be measured
# Pts mentions the column number (x-axis) and row number (Y-axis) 
# represented by Xp and Yp later in the code respectively, which needs to 
# be the center of the subset for which deformation is required. 
# The points must not be on the edges of the image.
# q_0 is the initial guess for U and V. It must have two values only. If no
# initial guess can be provided then use [0 0]. 
# Initial guess must be in
# the range of subset size. The values on sides of images or where
# iteration process is terminated are not included in the final results.
    
    # Global variables to be used
    
    global subset_size
    global ref_image
    global Xp
    global Yp
    global def_interp
    global def_interp_x
    global def_interp_y
    
    # point)
    if (mod(subsetSize,2) == 0):
        error('Subset must be odd?')
    
    
    # Read in the images (default directory is current working directory)
    ref_image=matplotlib.image.imread(ref_img)
# DIC_NR_images.m:37
    def_image=matplotlib.image.imread(def_img)
# DIC_NR_images.m:38
    
    Y_size,X_size=size(ref_image,nargout=2)
# DIC_NR_images.m:41
    
#Should already be floats?
    #ref_image=double(ref_img_read)

# DIC_NR_images.m:44

#Should already be floats?
    #def_image=double(def_img_read)

# DIC_NR_images.m:45
    
    subset_size=copy(subsetSize)
# DIC_NR_images.m:48
    spline_order=6
# DIC_NR_images.m:49
    
    Max_num_iter=40
# DIC_NR_images.m:52
#Declare TOL (ADDED)
    TOL = [0, 0, 0]
    TOL[1]=10 ** (- 8)
# DIC_NR_images.m:53
    
    TOL[2]=10 ** (- 8) / 2
# DIC_NR_images.m:54
    
    #     result = zeros(length(pts(:,1)),12);
    
    # condition to check that point of interest is not close to edge. Point
# must away from edge greater than half of subset adding 15 to it to have
# range of initial guess accuracy.
    Xmin=round((subset_size / 2) + 15)
# DIC_NR_images.m:61
    Ymin=copy(Xmin)
# DIC_NR_images.m:62
    Xmax=round(X_size - ((subset_size / 2) + 15))
# DIC_NR_images.m:63
    Ymax=round(Y_size - ((subset_size / 2) + 15))
# DIC_NR_images.m:64
    Xp=copy(Xmin)
# DIC_NR_images.m:65
    Yp=copy(Ymin)
# DIC_NR_images.m:66
    if ((Xp < Xmin) or (Yp < Ymin) or (Xp > Xmax) or (Yp > Ymax)):
        error('Process terminated!!! First point of centre of subset is on the edge of the image. ')
    
    #_____________Automatic Initial Guess______________________
    
    # Automatic Initial Guess
    q_0=zeros(6,1)
# DIC_NR_images.m:74
    q_0[arange(1,2),1]=ini_guess
# DIC_NR_images.m:75
    # The initial guess must lie between -range to range in pixels
    range_=15
# DIC_NR_images.m:77
    
    u_check=arange((round(q_0[1]) - range_),(round(q_0[1]) + range_))
# DIC_NR_images.m:78
    v_check=arange((round(q_0[2]) - range_),(round(q_0[2]) + range_))
# DIC_NR_images.m:79
    # Define the intensities of the first reference subset
    subref=ref_image[arange(Yp - floor(subset_size / 2),Yp + floor(subset_size / 2), dtype=int),arange(Xp - floor(subset_size / 2),Xp + floor(subset_size / 2), dtype=int)]
# DIC_NR_images.m:82
    # Preallocate some matrix space
    sum_diff_sq=zeros(numel(u_check),numel(v_check))
# DIC_NR_images.m:85
    # Check every value of u and v and see where the best match occurs
    for iter1 in arange(1,numel(u_check)).reshape(-1):
        for iter2 in arange(1,numel(v_check)).reshape(-1):
            subdef=def_image[arange((Yp - floor(subset_size / 2) + v_check[iter2]),(Yp + floor(subset_size / 2) + v_check[iter2]), dtype=int),arange((Xp - floor(subset_size / 2) + u_check[iter1]),(Xp + floor(subset_size / 2) + u_check[iter1]), dtype=int)]
# DIC_NR_images.m:89
            sum_diff_sq[iter2,iter1]=sum(sum((subref - subdef) ** 2))
# DIC_NR_images.m:91
    
    TMP1,OFFSET1=min(min(sum_diff_sq,[],2),nargout=2) #Not implemented in libsmop
    '''
    M = min(A,[],dim) returns the minimum element along dimension dim. For example, if A is a matrix, then min(A,[],2) is a column vector containing the minimum value of each row.
    '''
# DIC_NR_images.m:94
    TMP2,OFFSET2=min(min(sum_diff_sq,[],1),nargout=2) #Not implemented in libsmop
# DIC_NR_images.m:95
    q_0[1]=u_check(OFFSET2)
# DIC_NR_images.m:96
    q_0[2]=v_check(OFFSET1)
# DIC_NR_images.m:97
    #clear('u_check','v_check','iter1','iter2','subref','subdef','sum_diff_sq','TMP1','TMP2','OFFSET1','OFFSET2')
    # Preallocate the matrix that holds the deformation parameter results
    DEFORMATION_PARAMETERS=zeros(Y_size,X_size,12)
# DIC_NR_images.m:102
    #Declare q_k (ADDED)
    q_k = numpy.zeros((6,1))
    # Set the initial guess to be the "last iteration's" solution.
    q_k[arange(1,6),1]=q_0(arange(1,6),1)
# DIC_NR_images.m:105
    #_______________COMPUTATIONS________________
    
    # Start the timer: Track the time it takes to perform the heaviest computations
    tic
    #__________FIT SPLINE ONTO DEFORMED SUBSET________________________
# Obtain the size of the reference image
    Y_size,X_size=size(ref_image,nargout=2)
# DIC_NR_images.m:114
    # Define the deformed image's coordinates
    X_defcoord=arange(1,X_size)
# DIC_NR_images.m:117
    Y_defcoord=arange(1,Y_size)
# DIC_NR_images.m:118
    # Fit the interpolating spline: g(x,y)
    #Need to find python equivalent...
    #def_interp=spapi(cellarray([spline_order,spline_order]),cellarray([Y_defcoord,X_defcoord]),def_image(Y_defcoord,X_defcoord))

    def_interp= BPoly.from_derivatives(cellarray([spline_order,spline_order]),cellarray([Y_defcoord,X_defcoord]),def_image(Y_defcoord,X_defcoord))
# DIC_NR_images.m:121
    # Find the partial derivitives of the spline: dg/dx and dg/dy
#def_interp_x = fnder(def_interp, [0,1]);
#def_interp_y = fnder(def_interp, [1,0]);
    
    # Convert all the splines from B-form into ppform to make it
# computationally cheaper to evaluate. Also find partial derivatives of
# spline w.r.t x and y

    '''
    Seriously have no idea for the following three functions
    '''
    #def_interp=fn2fm(def_interp,'pp')
# DIC_NR_images.m:130
    #def_interp_x=fnder(def_interp,concat([0,1]))
    def_interp_x=def_interp(concat([0,1]))
# DIC_NR_images.m:131
    #def_interp_y=fnder(def_interp,concat([1,0]))
    def_interp_y=def_interp(concat([1,0]))

# DIC_NR_images.m:132
    #_________________________________________________________________________
    t_interp=copy(toc)
# DIC_NR_images.m:134
    
    # MAIN CORRELATION LOOP -- CORRELATE THE POINTS REQUESTED
    
    # for i=1:length(pts(:,1))
    for yy in arange(Ymin,Ymax).reshape(-1):
        if yy > Ymin:
            q_k[arange(1,6),1]=DEFORMATION_PARAMETERS(yy - 1,Xmin,arange(1,6))
# DIC_NR_images.m:142
        for xx in arange(Xmin,Xmax).reshape(-1):
            #Points for correlation and initializaing the q matrix
            Xp=copy(xx)
# DIC_NR_images.m:147
            Yp=copy(yy)
# DIC_NR_images.m:148
            t_tmp=copy(toc)
# DIC_NR_images.m:149
            #         if (itr_skip == 0)
            # Initialize some values
            n=0
# DIC_NR_images.m:156
            C_last,GRAD_last,HESS=C_First_Order.C_First_Order(q_k,nargout=3)
# DIC_NR_images.m:157
            optim_completed=copy(false)
# DIC_NR_images.m:158
            
            #if isnan(abs(mean(mean(HESS)))):
            if (abs(numpy.mean(numpy.mean(HESS))) != 0):
                disp(yy)
                disp(xx)
                optim_completed=copy(true)
# DIC_NR_images.m:163
                #                 itr_skip = 1;
            while optim_completed == false:

                # Compute the next guess and update the values
                delta_q=numpy.linalg.solve(HESS,(- GRAD_last))
# DIC_NR_images.m:170
                q_k=q_k + delta_q
# DIC_NR_images.m:171
                C,GRAD,HESS=C_First_Order.C_First_Order(q_k,nargout=3)
# DIC_NR_images.m:172
                # Add one to the iteration counter
                n=n + 1
# DIC_NR_images.m:175
                # Check to see if the values have converged according to the stopping criteria
                if n > Max_num_iter or (abs(C - C_last) < TOL[1] and all(abs(delta_q) < TOL[2])):
                    optim_completed=copy(true)
# DIC_NR_images.m:179
                C_last=copy(C)
# DIC_NR_images.m:182
                GRAD_last=copy(GRAD)
# DIC_NR_images.m:183

            #_________________________________________________________________________
            t_optim=toc - t_tmp
# DIC_NR_images.m:187
            # Store the current displacements
            DEFORMATION_PARAMETERS[yy,xx,1]=q_k(1)
# DIC_NR_images.m:192
            DEFORMATION_PARAMETERS[yy,xx,2]=q_k(2)
# DIC_NR_images.m:193
            DEFORMATION_PARAMETERS[yy,xx,3]=q_k(3)
# DIC_NR_images.m:194
            DEFORMATION_PARAMETERS[yy,xx,4]=q_k(4)
# DIC_NR_images.m:195
            DEFORMATION_PARAMETERS[yy,xx,5]=q_k(5)
# DIC_NR_images.m:196
            DEFORMATION_PARAMETERS[yy,xx,6]=q_k(6)
# DIC_NR_images.m:197
            DEFORMATION_PARAMETERS[yy,xx,7]=1 - C
# DIC_NR_images.m:198
            # store points which are correlated in reference image i.e. center of subset
            DEFORMATION_PARAMETERS[yy,xx,8]=Xp
# DIC_NR_images.m:200
            DEFORMATION_PARAMETERS[yy,xx,9]=Yp
# DIC_NR_images.m:201
            DEFORMATION_PARAMETERS[yy,xx,10]=n
# DIC_NR_images.m:203
            DEFORMATION_PARAMETERS[yy,xx,11]=t_tmp
# DIC_NR_images.m:204
            DEFORMATION_PARAMETERS[yy,xx,12]=t_optim
# DIC_NR_images.m:205
        disp(yy)
        disp(xx)
    numpy.savetxt("Deformation_Parameters.csv", DEFORMATION_PARAMETERS, delimiter=",")
    return copy(DEFORMATION_PARAMETERS)
# DIC_NR_images.m:222

DIC_NR_images("ref500.bmp", "def500.bmp", 7, [0, 0])