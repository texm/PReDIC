import math
import numpy as np

def C_First_Order(q, globals):
	C = 0.0
	GRAD = 0.0
	HESS = 0.0

	u           = q[0]
	v           = q[1]
	du_dx       = q[2]
	dv_dy       = q[3]
	du_dy       = q[4]
	dv_dx       = q[5]

	subset_size = globals.subset_size
	ref_image = globals.ref_image
	Xp = globals.Xp
	Yp = globals.Yp

	#global subset_size;
	#global ref_image;
	#global Xp;
	#global Yp;
	#global def_interp;
	#global def_interp_x;
	#global def_interp_y;  

	i = np.arange(-math.floor(subset_size/2), floor(subset_size/2))
	j = np.arange(-math.floor(subset_size/2), floor(subset_size/2))

	I_matrix, J_matrix = np.meshgrid(i, j)

	N = np.multiply(subset_size, subset_size)

	I = np.reshape(I_matrix, 1, N)
	J = np.reshape(J_matrix, 1, N)

	X = Xp + u + I + np.multiply(I, du_dx) + np.multiply(J, du_dy)
	Y = Yp + v + J + np.multiply(J, dv_dy) + np.multiply(I, dv_dx)

	'''
	%-OBJECTIVE FUNCTION "C"---------------------------------------------------
		% f represents intensities of the discrete points in the ref subset
		f = reshape(ref_image(Yp+j, Xp+i), 1,N);
		
		% g represents the intensities of the continuous splined def sector
		g = fnval(def_interp, [Y;X]);
	   
		% The following represents the double sums of C, 
		%(The summation limits are from -floor(subset_size/2) to floor(subset_size/2)
		SS_f_g = sum(sum( ((f-g).^2) ));
		SS_f_sq = sum(sum( (f.^2) ));
		
		C = SS_f_g./SS_f_sq;
	%--------------------------------------------------------------------------
	 
	if nargout > 1
	%-GRADIENT OF "C"----------------------------------------------------------  
		% Evaluate the derivitives at the points of interest
		dg_dX = fnval(def_interp_x, [Y;X]);
		dg_dY = fnval(def_interp_y, [Y;X]);
		
		% Determine the derivitives of the coordinate terms (i.e. suppose that
		% the coordinates of "f" are f(X,Y) = f(Xp+i-u+..., Yp+j-v+...)
		dX_du = 1;
		dX_dv = 0;
		dX_dudx = I;
		dX_dvdy = 0;
		dX_dudy = J;
		dX_dvdx = 0;
		
		dY_du = 0;
		dY_dv = 1;
		dY_dudx = 0;
		dY_dvdy = J;
		dY_dudy = 0;
		dY_dvdx = I;

		
		% Express the chain rule for partial derivites on "g"
		dg_du   = dg_dX.*dX_du + dg_dY.*dY_du;
		dg_dv   = dg_dX.*dX_dv + dg_dY.*dY_dv;
		dg_dudx = dg_dX.*dX_dudx + dg_dY.*dY_dudx;
		dg_dvdy = dg_dX.*dX_dvdy + dg_dY.*dY_dvdy;
		dg_dudy = dg_dX.*dX_dudy + dg_dY.*dY_dudy;
		dg_dvdx = dg_dX.*dX_dvdx + dg_dY.*dY_dvdx;
		
		% Write out each value in the gradient vector
		dC_du = sum(sum( (g-f).*(dg_du) ));
		dC_dv = sum(sum( (g-f).*(dg_dv) ));
		dC_dudx = sum(sum( (g-f).*(dg_dudx) ));
		dC_dvdy = sum(sum( (g-f).*(dg_dvdy) ));
		dC_dudy = sum(sum( (g-f).*(dg_dudy) ));
		dC_dvdx = sum(sum( (g-f).*(dg_dvdx) ));
		
		GRAD = (2/SS_f_sq).*[ dC_du, dC_dv, dC_dudx, dC_dvdy, dC_dudy, dC_dvdx ]';
	%--------------------------------------------------------------------------

	if nargout > 2
	%-HESSIAN OF "C"-----------------------------------------------------------   

		% Write out each value in the Hessian Matrix (remember, it's symmetric,
		% so only half of the entries are need), using Knauss' approximation
		d2C_du2 = sum(sum( (dg_du).*(dg_du) ));               
		d2C_dv2 = sum(sum( (dg_dv).*(dg_dv) ));
		d2C_dudx2 = sum(sum( (dg_dudx).*(dg_dudx) ));
		d2C_dvdy2 = sum(sum( (dg_dvdy).*(dg_dvdy) ));
		d2C_dudy2 = sum(sum( (dg_dudy).*(dg_dudy) ));
		d2C_dvdx2 = sum(sum( (dg_dvdx).*(dg_dvdx) ));
		
		d2C_dudv = sum(sum( (dg_du).*(dg_dv) ));
		d2C_dududx = sum(sum( (dg_du).*(dg_dudx) ));
		d2C_dudvdy = sum(sum( (dg_du).*(dg_dvdy) ));
		d2C_dududy = sum(sum( (dg_du).*(dg_dudy) ));
		d2C_dudvdx = sum(sum( (dg_du).*(dg_dvdx) ));
		
		d2C_dvdudx = sum(sum( (dg_dv).*(dg_dudx) ));
		d2C_dvdvdy = sum(sum( (dg_dv).*(dg_dvdy) ));
		d2C_dvdudy = sum(sum( (dg_dv).*(dg_dudy) ));
		d2C_dvdvdx = sum(sum( (dg_dv).*(dg_dvdx) ));
		
		d2C_dudxdvdy = sum(sum( (dg_dudx).*(dg_dvdy) ));
		d2C_dudxdudy = sum(sum( (dg_dudx).*(dg_dudy) ));
		d2C_dudxdvdx = sum(sum( (dg_dudx).*(dg_dvdx) ));
		
		d2C_dvdydudy = sum(sum( (dg_dvdy).*(dg_dudy) ));
		d2C_dvdydvdx = sum(sum( (dg_dvdy).*(dg_dvdx) ));
		
		d2C_dudydvdx = sum(sum( (dg_dudy).*(dg_dvdx) ));
				 
				 
		HESS = (2/SS_f_sq).* [  d2C_du2,    d2C_dudv,   d2C_dududx,   d2C_dudvdy,   d2C_dududy,   d2C_dudvdx   ; ...
								d2C_dudv,   d2C_dv2,    d2C_dvdudx,   d2C_dvdvdy,   d2C_dvdudy,   d2C_dvdvdx   ; ...
								d2C_dududx, d2C_dvdudx, d2C_dudx2,    d2C_dudxdvdy, d2C_dudxdudy, d2C_dudxdvdx ; ...
								d2C_dudvdy, d2C_dvdvdy, d2C_dudxdvdy, d2C_dvdy2,    d2C_dvdydudy, d2C_dvdydvdx ; ...
								d2C_dududy, d2C_dvdudy, d2C_dudxdudy, d2C_dvdydudy, d2C_dudy2,    d2C_dudydvdx ; ...
								d2C_dudvdx, d2C_dvdvdx, d2C_dudxdvdx, d2C_dvdydvdx, d2C_dudydvdx, d2C_dvdx2   ];
	%--------------------------------------------------------------------------
	end % if nargout > 2
	end % if nargout > 1

	end % function
	'''

	return C, GRAD, HESS