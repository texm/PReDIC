import sys, os, unittest
PARENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(PARENT_DIR)
TEST_IMAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/testing_images/"

from deformation_measurement import DeformationMeasurer
from deformation_measurement import DIC_NR_images
from deformation_measurement import C_First_Order

from deformation_measurement.DIC_NR_images import *
#from test_generated import generate_images

import numpy as np
from PIL import Image
from scipy.interpolate import splrep, PPoly, RectBivariateSpline, BSpline, BPoly, bisplev, bisplrep, splder


class TestFunctions(unittest.TestCase):
	'''
	def test_import_successful(self):
		self.
		self.assertTrue(self.dm is not None)
	'''
	def test_interpolation(self):
		test_image_1 = np.array(Image.open(TEST_IMAGE_DIR + "ref50.bmp").convert('LA')) # numpy.array
		print(test_image_1.shape)
		print(test_image_1)
		test_image_b = test_image_1.astype('d')
		print(test_image_b.shape)
		print(test_image_b)

		X_size, Y_size, _tmp= test_image_b.shape
		print(X_size, Y_size, _tmp)

		col1 = test_image_b[48,0,0]
		col2 = test_image_b[49,0,0]

		test_image_c = test_image_b[:,:,0]
		print(test_image_c)

		Y_size, X_size,tmp = test_image_b.shape

		X_defcoord = np.arange(0, X_size, dtype=int) # Maybe zero?
		Y_defcoord = np.arange(0, Y_size, dtype=int)

		interp = RectBivariateSpline(X_defcoord, Y_defcoord, test_image_b[:,:,0], kx=5, ky=5)
		result1 = interp.ev(48,0)
		result2 = interp.ev(48.5,0)
		result3 = interp.ev(49,0)

		interp_res = np.empty([50,50])

		for y in range(50):
			for x in range(50):
				interp_res[x][y] = interp.ev(x,y)

		#savetxt_compact("image_actual", test_image_c)
		#savetxt_compact("deform", interp_res)

		print(result1)
		print(result2)
		print(result3)

		print(col1)
		print(col2)

	def test_initial_guess(self):

		test_image_1 = np.array(Image.open(TEST_IMAGE_DIR + "ref50.bmp").convert('LA')) # numpy.array
		test_image_1 = test_image_1.astype('d')

		q_k = initial_guess(test_image_1, test_image_1, [0,0], 15, 25, 25)
		#should be all 0
		print(q_k)

		a = np.array([[1,2],[3,4]])
		b = np.array([[1,1],[1,1]])
		print(np.square(a-b))
		print(np.sum(np.square(a-b)))
		print(np.sum(np.sum(np.square(a-b))))

	def test_fit_spline(self):

		#self.a = DIC_NR_images()

		test_image_1 = np.array(Image.open(TEST_IMAGE_DIR + "ref50.bmp").convert('LA')) # numpy.array
		test_image_1 = test_image_1.astype('d')

		actual_val_48_0 = test_image_1[48,0,0]
		actual_val_49_0 = test_image_1[49,0,0]

		#Note: this is using same image as ref and def
		interp, interp_x, interp_y = fit_spline(test_image_1, test_image_1, 5)
		result1 = interp.ev(48,0)
		result2 = interp.ev(48.5,0)
		result3 = interp.ev(49,0)

		print("Actual values at x? 48,49 y:0")
		print(actual_val_48_0)
		print(actual_val_49_0)

		print("Interpolated values at x? 48, 48.5, 49 y:0")
		print(result1)
		print(result2)
		print(result3)

    

class TestCase(unittest.TestCase):
#	def setUp(self):
#		self.dm = DeformationMeasurer()


	def test_import_successful(self):
		self.assertTrue(self.dm is not None)

	def test_interpolation(self):
		test_image_1 = np.array(Image.open("test_image_1.bmp").convert('LA')) # numpy.array

	'''
	def test_DIC_NR_images(self):
		matlab_output = [] # open csv file of matlab results
		python_output = self.dm.DIC_NR_images("ref50.bmp", "def50.bmp", 7, [0, 0])
		self.assertEqual(python_output, matlab_output)

	# cant really test without ^ finished
	def test_C_First_Order(self):
		matlab_output = (0.0, 0.0, 0.0) # save matlab results as json or something?
		python_output = matlab_output #self.dm.C_First_Order([0, 0, 0])
		self.assertEqual(python_output, matlab_output)
	'''
	'''
	def test_generated_images(self):
		ref_i, def_i = generate_images()
		self.assertIsNotNone(ref_i)
		self.assertIsNotNone(def_i)
	'''

def savetxt_compact(fname, x, fmt="%.6g", delimiter=','):
    with open(f"compact_{fname}.csv", 'w+') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

if __name__ == '__main__':
	unittest.main()
