from math import floor
import numpy as np

class C_First_Order(object):
	def set_image(self, ref_image, subset_size):
		self.ref_image    = ref_image
		self.subset_size  = subset_size


	def set_splines(self, def_interp, def_interp_x, def_interp_y):
		self.def_interp   = def_interp
		self.def_interp_x = def_interp_x
		self.def_interp_y = def_interp_y


	def define_deformed_subset(self, q, Xp, Yp):
		half_subset = floor(self.subset_size / 2)

		i = np.arange(-half_subset, half_subset + 1, dtype=int)
		j = np.arange(-half_subset, half_subset + 1, dtype=int)

		self.I_matrix, self.J_matrix = np.meshgrid(i, j)

		self.I = self.I_matrix.flatten()
		self.J = self.J_matrix.flatten()

		u           = q[0]
		v           = q[1]
		du_dx       = q[2]
		dv_dy       = q[3]
		du_dy       = q[4]
		dv_dx       = q[5]

		self.X = Xp + u + self.I + np.multiply(self.I, du_dx) + np.multiply(self.J, du_dy)
		self.Y = Yp + v + self.J + np.multiply(self.J, dv_dy) + np.multiply(self.I, dv_dx)


	def calculate(self, q, Xp, Yp, nargout=3):
		C = 0.0
		GRAD = 0.0
		HESS = 0.0

		half_subset = floor(self.subset_size / 2)

		self.define_deformed_subset(q, Xp, Yp)

		g = self.def_interp.ev(self.Y, self.X)

		y0 = Yp - half_subset
		y1 = Yp + half_subset+1

		x0 = Xp - half_subset
		x1 = Xp + half_subset+1

		reference_subset = self.ref_image[y0:y1, x0:x1, 0]
		f = reference_subset.flatten()

		SS_f_g = np.sum(np.sum(np.square((f-g))))
		SS_f_sq = np.sum(np.sum(np.square(f)))

		if(SS_f_sq == 0):
			raise error("Reference subset contains no image data, a larger subset is necessary to capture more speckle information.")

		C = np.divide(SS_f_g, SS_f_sq)

		if nargout > 1:

			dg_dX = self.def_interp.ev(self.Y, self.X, 0, 1)
			dg_dY = self.def_interp.ev(self.Y, self.X, 1, 0)

			dX_du = 1
			dX_dv = 0
			dX_dudx = self.I
			dX_dudy = self.J
			dX_dvdx = 0
			dX_dvdy = 0

			dY_du = 0
			dY_dv = 1
			dY_dudx = 0
			dY_dudy = 0
			dY_dvdy = self.J
			dY_dvdx = self.I

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

			GRAD = np.multiply(2/SS_f_sq, np.array([dC_du, dC_dv, dC_dudx, dC_dvdy, dC_dudy, dC_dvdx]))

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