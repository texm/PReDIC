import math
import numpy as np
import scipy as sp

def C_First_Order(q, G, nargout=2):
	C = 0.0
	GRAD = 0.0
	HESS = 0.0

	u           = q[0]
	v           = q[1]
	du_dx       = q[2]
	dv_dy       = q[3]
	du_dy       = q[4]
	dv_dx       = q[5]

	subset_size = G.subset_size
	ref_image = G.ref_image
	Xp = G.Xp
	Yp = G.Yp
	def_interp = G.def_interp
	def_interp_x = G.def_interp_x
	def_interp_y = G.def_interp_y

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

	f = np.reshape(ref_image[Yp + j, Xp + i], 1, N)
	g = def_interp([[Y], [X]])

	SS_f_g = np.sum(np.sum(np.power((f-g), 2)))
	SS_f_sq = np.sum(np.sum(np.power(f, 2)))

	C = np.divide(SS_f_g, SS_f_sq)

	if nargout > 1:
		dg_dX = def_interp_x([[Y], [X]])
		dg_dY = def_interp_x([[Y], [X]])

		dX_du = 1
		dX_dv = 0
		dX_dudx = I
		dX_dvdy = 0
		dX_dudy = J
		dX_dvdx = 0
		
		dY_du = 0
		dY_dv = 1
		dY_dudx = 0
		dY_dvdy = J
		dY_dudy = 0
		dY_dvdx = I

		dg_du = np.multiply(dg_dX, dX_du) + np.multiply(dg_dY, dY_du)
		dg_dv = np.multiply(dg_dX, dX_dv) + np.multiply(dg_dY, dY_dv)
		dg_dudx = np.multiply(dg_dX, dX_dudx) + np.multiply(dg_dY, dY_dudx)
		dg_dvdy = np.multiply(dg_dX, dX_dvdy) + np.multiply(dg_dY, dY_dvdy)
		dg_dudy = np.multiply(dg_dX, dX_dudy) + np.multiply(dg_dY, dY_dudy)
		dg_dvdx = np.multiply(dg_dX, dX_dxdx) + np.multiply(dg_dY, dY_dvdx)

		dC_du = np.sum(np.sum(np.multiply(g-f, dg_du)))
		dC_dv = np.sum(np.sum(np.multiply(g-f, dg_dv)))
		dC_dudx = np.sum(np.sum(np.multiply(g-f, dg_dudx)))
		dC_dvdy = np.sum(np.sum(np.multiply(g-f, dg_dvdy)))
		dC_dudy = np.sum(np.sum(np.multiply(g-f, dg_dudy)))
		dC_dvdx = np.sum(np.sum(np.multiply(g-f, dg_dvdx)))

		GRAD = np.multiply(2/SS_f_sq, np.array([ dC_du, dC_dv, dC_dudx, dC_dvdy, dC_dudy, dC_dvdx ]))

	if nargout > 2:
		d2C_du2 = np.sum(np.sum(np.multiply(dg_du, dg_du)))               
		d2C_dv2 = np.sum(np.sum(np.multiply(dg_dv, dg_dv)))
		d2C_dudx2 = np.sum(np.sum(np.multiply(dg_dudx, dg_dudx)))
		d2C_dvdy2 = np.sum(np.sum(np.multiply(dg_dvdy, dg_dvdy)))
		d2C_dudy2 = np.sum(np.sum(np.multiply(dg_dudy, dg_dudy)))
		d2C_dvdx2 = np.sum(np.sum(np.multiply(dg_dvdx, dg_dvdx)))
		
		d2C_dudv = np.sum(np.sum(np.multiply(dg_du, dg_dv)))
		d2C_dududx = np.sum(np.sum(np.multiply(dg_du, dg_dudx)))
		d2C_dudvdy = np.sum(np.sum(np.multiply(dg_du, dg_dvdy)))
		d2C_dududy = np.sum(np.sum(np.multiply(dg_du, dg_dudy)))
		d2C_dudvdx = np.sum(np.sum(np.multiply(dg_du, dg_dvdx)))
		
		d2C_dvdudx = np.sum(np.sum(np.multiply(dg_dv, dg_dudx)))
		d2C_dvdvdy = np.sum(np.sum(np.multiply(dg_dv, dg_dvdy)))
		d2C_dvdudy = np.sum(np.sum(np.multiply(dg_dv, dg_dudy)))
		d2C_dvdvdx = np.sum(np.sum(np.multiply(dg_dv, dg_dvdx)))
		
		d2C_dudxdvdy = np.sum(np.sum(np.multiply(dg_dudx, dg_dvdy)))
		d2C_dudxdudy = np.sum(np.sum(np.multiply(dg_dudx, dg_dudy)))
		d2C_dudxdvdx = np.sum(np.sum(np.multiply(dg_dudx, dg_dvdx)))
		
		d2C_dvdydudy = np.sum(np.sum(np.multiply(dg_dvdy, dg_dudy)))
		d2C_dvdydvdx = np.sum(np.sum(np.multiply(dg_dvdy, dg_dvdx)))
		
		d2C_dudydvdx = np.sum(np.sum(np.multiply(dg_dudy, dg_dvdx)))
				 
		marr = np.array([ \
				[d2C_du2, d2C_dudv, d2C_dududx, d2C_dudvdy, d2C_dududy, d2C_dudvdx], \
				[d2C_dudv, d2C_dv2, d2C_dvdudx, d2C_dvdvdy, d2C_dvdudy, d2C_dvdvdx], \
				[d2C_dududx, d2C_dvdudx, d2C_dudx2,    d2C_dudxdvdy, d2C_dudxdudy, d2C_dudxdvdx], \
				[d2C_dudvdy, d2C_dvdvdy, d2C_dudxdvdy, d2C_dvdy2,    d2C_dvdydudy, d2C_dvdydvdx] \
				[d2C_dududy, d2C_dvdudy, d2C_dudxdudy, d2C_dvdydudy, d2C_dudy2,    d2C_dudydvdx] \
				[d2C_dudvdx, d2C_dvdvdx, d2C_dudxdvdx, d2C_dvdydvdx, d2C_dudydvdx, d2C_dvdx2] \
			])
		HESS = np.multiply(2/SS_f_sq, marr)
	'''	 
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