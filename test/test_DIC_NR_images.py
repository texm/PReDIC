import sys, os, unittest, pytest

PARENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(PARENT_DIR)

TEST_IMAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/testing_images/"

from predic import DIC_NR

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline
import math

class Test_DIC_NR(unittest.TestCase):

	def test_set_parameters(self):
		#TEST 1
		#Image too small for subset size and buffer room of 15.
		ref_img = np.expand_dims(np.reshape(np.arange(40*40), (40,40)), axis = 2)
		ref_img = np.insert(ref_img, 1, 0, axis = 2)
		def_img = np.roll(ref_img.copy(), 1, axis = 1)

		dicr_1 = DIC_NR()

		with pytest.raises(ValueError):
			dicr_1.set_parameters(ref_img, def_img, 11, [0,0])

		#TEST 2
		#Image size okay.
		ref_img = np.expand_dims(np.reshape(np.arange(41*41), (41,41)), axis = 2)
		ref_img = np.insert(ref_img, 1, 0, axis = 2)
		def_img = np.roll(ref_img.copy(), 1, axis = 1)

		dicr_1 = DIC_NR()
		dicr_1.set_parameters(ref_img, def_img, 11, [0,0])

		self.assertEqual(dicr_1.Xmin, 20)
		self.assertEqual(dicr_1.Ymin, 20)
		self.assertEqual(dicr_1.Xmax, 20)
		self.assertEqual(dicr_1.Ymax, 20)
		self.assertEqual(dicr_1.Xp, 20)
		self.assertEqual(dicr_1.Xp, 20)

	def test_initial_guess(self):
		#TEST 1
		#moving "image" all to right by 1, u = 1, v = 0
		#check the logic that u & v are indexed correctly
		ref_img = np.expand_dims(np.reshape(np.arange(41*41), (41,41)), axis = 2)
		ref_img = np.insert(ref_img, 1, 0, axis = 2)
		def_img = np.roll(ref_img.copy(), 1, axis = 1)

		dicr_1 = DIC_NR()
		dicr_1.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "def50.bmp", 11, [0,0])
		dicr_1.initial_guess(ref_img, def_img)

		self.assertEqual(dicr_1.q_k[0], 1)
		self.assertEqual(dicr_1.q_k[1], 0)

		#TEST 2
		#moving "image" all down by 1, u = 0, v = 1
		#check the logic that u & v are indexed correctly
		ref_img = np.expand_dims(np.reshape(np.arange(41*41), (41,41)), axis = 2)
		ref_img = np.insert(ref_img, 1, 0, axis = 2)
		def_img = np.roll(ref_img.copy(), 1, axis = 0)

		dicr_1 = DIC_NR()
		dicr_1.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "def50.bmp", 11, [0,0])
		dicr_1.initial_guess(ref_img, def_img)

		self.assertEqual(dicr_1.q_k[0], 0)
		self.assertEqual(dicr_1.q_k[1], 1)

		#TEST 3
		#comparing the same image with itself, u & v should equal 0
		dicr = DIC_NR()
		dicr.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "ref50.bmp", 11, [0,0])
		dicr.initial_guess()

		self.assertEqual(dicr.q_k[0], 0)
		self.assertEqual(dicr.q_k[1], 0)

		#TEST 4
		#comparing test images, u=2, v = 0
		dicr = DIC_NR()
		dicr.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "def50.bmp", 11, [0,0])
		dicr.initial_guess()

		self.assertEqual(dicr.q_k[0], 2)
		self.assertEqual(dicr.q_k[1], 0)
		

	def test_fit_spline(self):

		test_image_1 = np.array(Image.open(TEST_IMAGE_DIR + "ref50.bmp").convert('LA')) # numpy.array
		test_image_1 = test_image_1.astype('d')

		actual_val_48_0 = test_image_1[48,0,0]
		actual_val_49_0 = test_image_1[49,0,0]

		#Note: this is using same image as ref and def
		dicnr = DIC_NR()
		dicnr.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "ref50.bmp", 11, [0,0])
		dicnr.initial_guess()
		dicnr.fit_spline()

		result1 = dicnr.def_interp.ev(48,0)
		result2 = dicnr.def_interp.ev(48.5,0)
		result3 = dicnr.def_interp.ev(49,0)

		self.assertTrue(math.isclose(actual_val_48_0, result1, rel_tol = 0.01))
		self.assertTrue(actual_val_48_0<= result2 <= actual_val_49_0)
		self.assertTrue(math.isclose(actual_val_49_0, result3, rel_tol = 0.01))

		#TODO: test derivative spline

	def test_whole(self):
		dic_2 = DIC_NR()
		dic_2.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "def50.bmp",11,[0,0])
		
		output_2 = dic_2.calculate()

		self.assertEqual(output_2[20,20,0], 2)
		self.assertTrue(math.isclose(output_2[20,20,1], 0.0, abs_tol = 0.01))
