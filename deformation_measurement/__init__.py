from deformation_measurement.C_First_Order import C_First_Order
from deformation_measurement.DIC_NR_images import DIC_NR_images

class DeformationMeasurer(object):
	def __init__(self):
		super(DeformationMeasurer, self).__init__()

	def DIC_NR_images(self, ref_img, def_img, subsetSize, ini_guess):
		return DIC_NR_images(ref_img, def_img, subsetSize, ini_guess)

	def C_First_Order(self, q):
		return C_First_Order(q)