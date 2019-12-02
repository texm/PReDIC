import sys, os, unittest, pytest

PARENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(PARENT_DIR)

TEST_IMAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/testing_images/"

from predic import DIC_NR, C_First_Order

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline

class Test_C_First_Order(unittest.TestCase):

	def test_define_deformed_subset(self):
		ref_img = np.expand_dims(np.reshape(np.arange(40*40), (40,40)), axis = 2)
		ref_img = np.insert(ref_img, 1, 0, axis = 2)

		q = [1,0,0,0,0,0]

		cfo = C_First_Order()
		cfo.set_image(ref_img, 11)
		cfo.define_deformed_subset(q, 20,20)

		#Center of deformed subset
		#Moved from (20,20) to (21,20) according to u=1
		self.assertEqual(cfo.X[60] , 21)
		self.assertEqual(cfo.Y[60] , 20)

	def test_calculate(self):
		ref_img = np.expand_dims(np.reshape(np.arange(41*41), (41,41)), axis = 2)
		ref_img = np.insert(ref_img, 1, 0, axis = 2)
		def_img = np.roll(ref_img.copy(), 1, axis = 1)
		
		dicr_1 = DIC_NR()
		dicr_1.set_parameters(ref_img, def_img, 11, [0,0])

		q = [1,0,0,0,0,0]

		cfo = C_First_Order()
		cfo.set_image(ref_img, 11)
		cfo.set_splines(dicr_1.def_interp, dicr_1.def_interp_x, dicr_1.def_interp_y)
		a, b, c = cfo.calculate(q, 20, 20)

	def test_calculate2(self):
		ref_img = np.expand_dims(np.reshape(np.arange(41*41), (41,41)), axis = 2)
		ref_img = np.multiply(ref_img, 0.1*ref_img)
		ref_img = np.insert(ref_img, 1, 0, axis = 2)
		def_img = np.roll(ref_img.copy(), 1, axis = 1)
		
		dicr_1 = DIC_NR()
		dicr_1.set_parameters(ref_img, def_img, 11, [0,0])

		q = [1,0,0,0,0,0]

		cfo = C_First_Order()
		cfo.set_image(ref_img, 11)
		cfo.set_splines(dicr_1.def_interp, dicr_1.def_interp_x, dicr_1.def_interp_y)
		a, b, c = cfo.calculate(q, 20, 20)
