from math import floor
import numpy as np
import scipy as sp

def C_First_Order(q, _G, nargout=2):
	C = 0.0
	GRAD = 0.0
	HESS = 0.0

	u           = q[0]
	v           = q[1]
	du_dx       = q[2]
	dv_dy       = q[3]
	du_dy       = q[4]
	dv_dx       = q[5]

	subset_size = _G["subset_size"]
	ref_image = _G["ref_image"]
	Xp = _G["Xp"]
	Yp = _G["Yp"]
	def_interp = _G["def_interp"]
	def_interp_x = _G["def_interp_x"]
	def_interp_y = _G["def_interp_y"]

	i = np.arange(-floor(subset_size/2), floor(subset_size/2) + 1)
	j = np.arange(-floor(subset_size/2), floor(subset_size/2) + 1)

	I_matrix, J_matrix = np.meshgrid(i, j)

	N = np.multiply(subset_size, subset_size)

	I = np.reshape(I_matrix, (1, N))
	J = np.reshape(J_matrix, (1, N))

	X = Xp + u + I + np.multiply(I, du_dx) + np.multiply(J, du_dy)
	Y = Yp + v + J + np.multiply(J, dv_dy) + np.multiply(I, dv_dx)

	# TODO: why is ref_image[Yp +j, Xp + i] not len 50? (size 14)
	f = np.reshape(ref_image[Yp + j, Xp + i], (1, N))
	g = def_interp(Y, X)

	SS_f_g = np.sum(np.sum(np.power((f-g), 2)))
	SS_f_sq = np.sum(np.sum(np.power(f, 2)))

	C = np.divide(SS_f_g, SS_f_sq)

	if nargout > 1:
		dg_dX = def_interp_x(Y, X)
		dg_dY = def_interp_x(Y, X)

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
				 
		marr = np.array([
				[d2C_du2, d2C_dudv, d2C_dududx, d2C_dudvdy, d2C_dududy, d2C_dudvdx],
				[d2C_dudv, d2C_dv2, d2C_dvdudx, d2C_dvdvdy, d2C_dvdudy, d2C_dvdvdx],
				[d2C_dududx, d2C_dvdudx, d2C_dudx2,    d2C_dudxdvdy, d2C_dudxdudy, d2C_dudxdvdx],
				[d2C_dudvdy, d2C_dvdvdy, d2C_dudxdvdy, d2C_dvdy2,    d2C_dvdydudy, d2C_dvdydvdx],
				[d2C_dududy, d2C_dvdudy, d2C_dudxdudy, d2C_dvdydudy, d2C_dudy2,    d2C_dudydvdx],
				[d2C_dudvdx, d2C_dvdvdx, d2C_dudxdvdx, d2C_dvdydvdx, d2C_dudydvdx, d2C_dvdx2]
			])
		HESS = np.multiply(2/SS_f_sq, marr)

	return C, GRAD, HESS