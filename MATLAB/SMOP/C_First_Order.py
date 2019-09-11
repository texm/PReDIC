# Generated with SMOP  0.41
from smop.libsmop import *
# C_First_Order.m
import DIC_NR_images
    
    
@function
def C_First_Order(q=None,*args,**kwargs):
    varargin = C_First_Order.varargin
    nargin = C_First_Order.nargin

    # q is the vector of deformation variables, rename them for clarity
    u=q(1)
# C_First_Order.m:5
    v=q(2)
# C_First_Order.m:6
    du_dx=q(3)
# C_First_Order.m:7
    dv_dy=q(4)
# C_First_Order.m:8
    du_dy=q(5)
# C_First_Order.m:9
    dv_dx=q(6)
# C_First_Order.m:1
    '''
    global subset_size
    global DIC_NR_images.ref_image
    global DIC_NR_images.Xp
    global DIC_NR_images.Yp
    global DIC_NR_images.def_interp
    global DIC_NR_images.def_interp_x
    global DIC_NR_images.def_interp_y
    ''' 
      
    i=arange(- floor(DIC_NR_images.subset_size / 2),floor(DIC_NR_images.subset_size / 2),1)
# C_First_Order.m:22
    j=arange(- floor(DIC_NR_images.subset_size / 2),floor(DIC_NR_images.subset_size / 2),1)
# C_First_Order.m:23
    
    I_matrix,J_matrix=numpy.meshgrid(i,j,nargout=2)
# C_First_Order.m:26
    
    N=multiply(DIC_NR_images.subset_size,DIC_NR_images.subset_size)
# C_First_Order.m:29
    
    I=numpy.reshape(I_matrix,1,N)
# C_First_Order.m:32
    J=numpy.reshape(J_matrix,1,N)
# C_First_Order.m:33
    
    # every x coordinate in vector X and every corresponding y in vector Y
    X=DIC_NR_images.Xp + u + I + multiply(I,du_dx) + multiply(J,du_dy)
# C_First_Order.m:37
    Y=DIC_NR_images.Yp + v + J + multiply(J,dv_dy) + multiply(I,dv_dx)
# C_First_Order.m:38
    #-OBJECTIVE FUNCTION "C"---------------------------------------------------
    # f represents intensities of the discrete points in the ref subset
    f=numpy.reshape(DIC_NR_images.ref_image(DIC_NR_images.Yp + j,DIC_NR_images.Xp + i),1,N)
# C_First_Order.m:43
    
    #g=fnval(DIC_NR_images.def_interp,concat([[Y],[X]]))
    #EDITED 
    g=DIC_NR_images.def_interp(concat([[Y],[X]]))
# C_First_Order.m:46
    
    #(The summation limits are from -floor(DIC_NR_images.subset_size/2) to floor(DIC_NR_images.subset_size/2)
    SS_f_g=sum(sum(((f - g) ** 2)))
# C_First_Order.m:50
    SS_f_sq=sum(sum((f ** 2)))
# C_First_Order.m:51
    C=SS_f_g / SS_f_sq
# C_First_Order.m:53
    #--------------------------------------------------------------------------
    #Added
    if kwargs.get('nargout', -1) > 1:
        #-GRADIENT OF "C"----------------------------------------------------------  
    # Evaluate the derivitives at the points of interest
        #dg_dX=fnval(DIC_NR_images.def_interp_x,concat([[Y],[X]]))
        dg_dX=DIC_NR_images.def_interp_x(concat([[Y],[X]]))
# C_First_Order.m:59
        #dg_dY=fnval(DIC_NR_images.def_interp_y,concat([[Y],[X]]))
        dg_dY=DIC_NR_images.def_interp_y(concat([[Y],[X]]))
# C_First_Order.m:60
        # the coordinates of "f" are f(X,Y) = f(DIC_NR_images.Xp+i-u+..., DIC_NR_images.Yp+j-v+...)
        dX_du=1
# C_First_Order.m:64
        dX_dv=0
# C_First_Order.m:65
        dX_dudx=copy(I)
# C_First_Order.m:66
        dX_dvdy=0
# C_First_Order.m:67
        dX_dudy=copy(J)
# C_First_Order.m:68
        dX_dvdx=0
# C_First_Order.m:69
        dY_du=0
# C_First_Order.m:71
        dY_dv=1
# C_First_Order.m:72
        dY_dudx=0
# C_First_Order.m:73
        dY_dvdy=copy(J)
# C_First_Order.m:74
        dY_dudy=0
# C_First_Order.m:75
        dY_dvdx=copy(I)
# C_First_Order.m:76
        dg_du=multiply(dg_dX,dX_du) + multiply(dg_dY,dY_du)
# C_First_Order.m:80
        dg_dv=multiply(dg_dX,dX_dv) + multiply(dg_dY,dY_dv)
# C_First_Order.m:81
        dg_dudx=multiply(dg_dX,dX_dudx) + multiply(dg_dY,dY_dudx)
# C_First_Order.m:82
        dg_dvdy=multiply(dg_dX,dX_dvdy) + multiply(dg_dY,dY_dvdy)
# C_First_Order.m:83
        dg_dudy=multiply(dg_dX,dX_dudy) + multiply(dg_dY,dY_dudy)
# C_First_Order.m:84
        dg_dvdx=multiply(dg_dX,dX_dvdx) + multiply(dg_dY,dY_dvdx)
# C_First_Order.m:85
        dC_du=sum(sum(multiply((g - f),(dg_du))))
# C_First_Order.m:88
        dC_dv=sum(sum(multiply((g - f),(dg_dv))))
# C_First_Order.m:89
        dC_dudx=sum(sum(multiply((g - f),(dg_dudx))))
# C_First_Order.m:90
        dC_dvdy=sum(sum(multiply((g - f),(dg_dvdy))))
# C_First_Order.m:91
        dC_dudy=sum(sum(multiply((g - f),(dg_dudy))))
# C_First_Order.m:92
        dC_dvdx=sum(sum(multiply((g - f),(dg_dvdx))))
# C_First_Order.m:93
        GRAD=multiply((2 / SS_f_sq),concat([dC_du,dC_dv,dC_dudx,dC_dvdy,dC_dudy,dC_dvdx]).T)
# C_First_Order.m:95
        #--------------------------------------------------------------------------
        if kwargs.get('nargout', -1) > 2:
            #-HESSIAN OF "C"-----------------------------------------------------------
            # Write out each value in the Hessian Matrix (remember, it's symmetric,
    # so only half of the entries are need), using Knauss' approximation
            d2C_du2=sum(sum(multiply((dg_du),(dg_du))))
# C_First_Order.m:103
            d2C_dv2=sum(sum(multiply((dg_dv),(dg_dv))))
# C_First_Order.m:104
            d2C_dudx2=sum(sum(multiply((dg_dudx),(dg_dudx))))
# C_First_Order.m:105
            d2C_dvdy2=sum(sum(multiply((dg_dvdy),(dg_dvdy))))
# C_First_Order.m:106
            d2C_dudy2=sum(sum(multiply((dg_dudy),(dg_dudy))))
# C_First_Order.m:107
            d2C_dvdx2=sum(sum(multiply((dg_dvdx),(dg_dvdx))))
# C_First_Order.m:108
            d2C_dudv=sum(sum(multiply((dg_du),(dg_dv))))
# C_First_Order.m:110
            d2C_dududx=sum(sum(multiply((dg_du),(dg_dudx))))
# C_First_Order.m:111
            d2C_dudvdy=sum(sum(multiply((dg_du),(dg_dvdy))))
# C_First_Order.m:112
            d2C_dududy=sum(sum(multiply((dg_du),(dg_dudy))))
# C_First_Order.m:113
            d2C_dudvdx=sum(sum(multiply((dg_du),(dg_dvdx))))
# C_First_Order.m:114
            d2C_dvdudx=sum(sum(multiply((dg_dv),(dg_dudx))))
# C_First_Order.m:116
            d2C_dvdvdy=sum(sum(multiply((dg_dv),(dg_dvdy))))
# C_First_Order.m:117
            d2C_dvdudy=sum(sum(multiply((dg_dv),(dg_dudy))))
# C_First_Order.m:118
            d2C_dvdvdx=sum(sum(multiply((dg_dv),(dg_dvdx))))
# C_First_Order.m:119
            d2C_dudxdvdy=sum(sum(multiply((dg_dudx),(dg_dvdy))))
# C_First_Order.m:121
            d2C_dudxdudy=sum(sum(multiply((dg_dudx),(dg_dudy))))
# C_First_Order.m:122
            d2C_dudxdvdx=sum(sum(multiply((dg_dudx),(dg_dvdx))))
# C_First_Order.m:123
            d2C_dvdydudy=sum(sum(multiply((dg_dvdy),(dg_dudy))))
# C_First_Order.m:125
            d2C_dvdydvdx=sum(sum(multiply((dg_dvdy),(dg_dvdx))))
# C_First_Order.m:126
            d2C_dudydvdx=sum(sum(multiply((dg_dudy),(dg_dvdx))))
# C_First_Order.m:128
            HESS=multiply((2 / SS_f_sq),concat([[d2C_du2,d2C_dudv,d2C_dududx,d2C_dudvdy,d2C_dududy,d2C_dudvdx],[d2C_dudv,d2C_dv2,d2C_dvdudx,d2C_dvdvdy,d2C_dvdudy,d2C_dvdvdx],[d2C_dududx,d2C_dvdudx,d2C_dudx2,d2C_dudxdvdy,d2C_dudxdudy,d2C_dudxdvdx],[d2C_dudvdy,d2C_dvdvdy,d2C_dudxdvdy,d2C_dvdy2,d2C_dvdydudy,d2C_dvdydvdx],[d2C_dududy,d2C_dvdudy,d2C_dudxdudy,d2C_dvdydudy,d2C_dudy2,d2C_dudydvdx],[d2C_dudvdx,d2C_dvdvdx,d2C_dudxdvdx,d2C_dvdydvdx,d2C_dudydvdx,d2C_dvdx2]]))
# C_First_Order.m:131
            #--------------------------------------------------------------------------
    
    return C,GRAD,HESS
    
if __name__ == '__main__':
    pass
    