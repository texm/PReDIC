from math import floor
import numpy as np
import scipy as sp
import deformation_measurement.Globs as Globs
#import Globs

def ev_concatenate(def_interp, X,Y,subset_size, xd=0, yd=0):
	N = subset_size * subset_size
	t = def_interp.ev(X,Y,dx=xd, dy=yd)
	g = np.zeros_like(t)
	tmp = 0
	for first_index in range(1,subset_size+1):
		for second_index in range(1,subset_size+1):
			g[0,tmp] = t[0,((second_index - 1)*7+first_index)-1]
			tmp+=1
	return g

def define_deformed_subset(subset_size, Xp, Yp, u, v, du_dx, du_dy, dv_dx, dv_dy):

	i = np.arange(-floor(subset_size/2), floor(subset_size/2)+1, dtype=int)
	j = np.arange(-floor(subset_size/2), floor(subset_size/2)+1, dtype=int)

	I_matrix, J_matrix = np.meshgrid(i, j)

	N = subset_size * subset_size
	I = np.reshape(I_matrix, (1, N), 'F')
	J = np.reshape(J_matrix, (1, N), 'F')

	X = Xp + u + I + np.multiply(I, du_dx) + np.multiply(J, du_dy)
	Y = Yp + v + J + np.multiply(J, dv_dy) + np.multiply(I, dv_dx)

	return i, j, I_matrix, J_matrix, N, I, J, X, Y


def C_First_Order(q, nargout=3):

	C = 0.0
	GRAD = 0.0
	HESS = 0.0

	u           = q[0]
	v           = q[1]
	du_dx       = q[2]
	dv_dy       = q[3]
	du_dy       = q[4]
	dv_dx       = q[5]

	subset_size = Globs.subset_size
	ref_image = Globs.ref_image
	Xp = Globs.Xp
	Yp = Globs.Yp
	def_interp = Globs.def_interp
	def_interp_x = Globs.def_interp_x
	def_interp_y = Globs.def_interp_y

	i = np.arange(-floor(subset_size/2), floor(subset_size/2)+1, dtype=int)
	j = np.arange(-floor(subset_size/2), floor(subset_size/2)+1, dtype=int)

	I_matrix, J_matrix = np.meshgrid(i, j)

	N = subset_size * subset_size
	I = np.reshape(I_matrix, (1, N), 'F')
	J = np.reshape(J_matrix, (1, N), 'F')

	X = Xp + u + I + np.multiply(I, du_dx) + np.multiply(J, du_dy)
	Y = Yp + v + J + np.multiply(J, dv_dy) + np.multiply(I, dv_dx)

	f = np.reshape(ref_image[(Yp + J_matrix - 1), (Xp + I_matrix - 1), 0], (1, N), 'F')
	#tmp = ref_image[(Yp + J_matrix), (Xp + I_matrix),0]
	#np.vstack((Y,X))
	t = def_interp.ev(X,Y)
	g = np.zeros_like(t)
	tmp = 0
	for first_index in range(1,subset_size+1):
		for second_index in range(1,subset_size+1):
			g[0,tmp] = t[0,((second_index - 1)*7+first_index)-1]
			tmp+=1
	#print(f)
	#print(g)
	temp = (f-g)
	#print(temp)

	SS_f_g = np.sum(np.sum(np.square((f-g))))
	SS_f_sq = np.sum(np.sum(np.square(f)))

	C = np.divide(SS_f_g, SS_f_sq)

	if nargout > 1:
		dg_dX = ev_concatenate(def_interp, X,Y,subset_size,0,1)
		dg_dY = ev_concatenate(def_interp, X,Y,subset_size,1,0)

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
		dg_dvdx = np.multiply(dg_dX, dX_dvdx) + np.multiply(dg_dY, dY_dvdx)

		dC_du = np.sum(np.sum(np.multiply((g-f), dg_du)))
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
				 
		marr = np.array([
				[d2C_du2, d2C_dudv, d2C_dududx, d2C_dudvdy, d2C_dududy, d2C_dudvdx],
				[d2C_dudv, d2C_dv2, d2C_dvdudx, d2C_dvdvdy, d2C_dvdudy, d2C_dvdvdx],
				[d2C_dududx, d2C_dvdudx, d2C_dudx2,    d2C_dudxdvdy, d2C_dudxdudy, d2C_dudxdvdx],
				[d2C_dudvdy, d2C_dvdvdy, d2C_dudxdvdy, d2C_dvdy2,    d2C_dvdydudy, d2C_dvdydvdx],
				[d2C_dududy, d2C_dvdudy, d2C_dudxdudy, d2C_dvdydudy, d2C_dudy2,    d2C_dudydvdx],
				[d2C_dudvdx, d2C_dvdvdx, d2C_dudxdvdx, d2C_dvdydvdx, d2C_dudydvdx, d2C_dvdx2]
			])
		HESS = np.multiply((2/SS_f_sq), marr)

	return C, GRAD, HESS
