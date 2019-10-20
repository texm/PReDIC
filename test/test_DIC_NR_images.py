import sys, os, unittest

PARENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(PARENT_DIR)

TEST_IMAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/testing_images/"

from deformation_measurement import DIC_NR

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline

class Test_DIC_NR(unittest.TestCase):

	def test_initial_guess(self):

		#TEST 1
		#moving "image" all to right by 1, u = 1, v = 0
		#check the logic that u & v are indexed correctly
		ref_img = np.expand_dims(np.reshape(np.arange(40*40), (40,40)), axis = 2)
		ref_img = np.insert(ref_img, 1, 0, axis = 2)
		def_img = np.roll(ref_img.copy(), 1, axis = 1)

		dicr_1 = DIC_NR()
		dicr_1.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "def50.bmp", 11, [0,0])
		dicr_1.initial_guess(ref_img, def_img)

		print(dicr_1.q_k)

		self.assertEqual(dicr_1.q_k[0], 1)
		self.assertEqual(dicr_1.q_k[1], 0)

		#TEST 2
		#comparing the same image with itself, u & v should equal 0
		dicr = DIC_NR()
		dicr.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "ref50.bmp", 11, [0,0])
		dicr.initial_guess(ref_img, ref_img)

		self.assertEqual(dicr.q_k[0], 0)
		self.assertEqual(dicr.q_k[1], 0)

	def test_fit_spline(self):

		test_image_1 = np.array(Image.open(TEST_IMAGE_DIR + "ref50.bmp").convert('LA')) # numpy.array
		test_image_1 = test_image_1.astype('d')

		actual_val_48_0 = test_image_1[48,0,0]
		actual_val_49_0 = test_image_1[49,0,0]

		#Note: this is using same image as ref and def
		dicnr = DIC_NR()
		dicnr.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "ref50.bmp", 11, [0,0])
		dicnr.fit_spline()

		result1 = dicnr.def_interp.ev(48,0)
		result2 = dicnr.def_interp.ev(48.5,0)
		result3 = dicnr.def_interp.ev(49,0)

		print("Actual values at x? 48,49 y:0")
		print(actual_val_48_0)
		print(actual_val_49_0)

		print("Interpolated values at x? 48, 48.5, 49 y:0")
		print(result1)
		print(result2)
		print(result3)

	def test_something(self):
		'''
		test_image_1 = np.array(Image.open(TEST_IMAGE_DIR + "ref50.bmp").convert('LA')) # numpy.array
		test_image_1 = test_image_1.astype('d')

		ref_img = np.expand_dims(np.reshape(np.arange(40*40), (40,40)), axis = 2)
		ref_img = np.insert(ref_img, 1, 0, axis = 2)
		def_img = np.roll(ref_img.copy(), 1, axis = 1)

		print("ref image:")
		print(ref_img[15:26,15:26,0])

		print("def image:")
		print(def_img[15:26,16:27,0])

		dic = DIC_NR(ref_img, def_img, 11, [0,0])
		dic.initial_guess()
		dic.fit_spline()

		print(dic.q_k)
		print(dic.Xp)
		print(dic.Yp)

		output = dic.calculate()

		#c_last, grad_last, hess = dic.cfo.calculate(dic.q_k, dic.Xp, dic.Yp)
		'''
		#print(output[20,20,0])
		#print(output[20,20,1])
		'''
		print("C")
		print(c_last)
		print("grad")
		print(grad_last)
		print("hess")
		print(hess)
		'''

		
		dic_2 = DIC_NR()
		dic_2.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "def50.bmp",11,[0,0])
		dic_2.initial_guess()
		dic_2.fit_spline()

		#c_last_2, grad_last_2, hess_2 = dic_2.cfo.calculate(dic_2.q_k, dic_2.Xp, dic_2.Yp)
		
		output_2 = dic_2.calculate()
		#print("other")
		print(output_2[20:30,20:30,0])
		print(output_2[20:30,20:30,1])

		x,y,z = output_2.shape
		sav = np.swapaxes(output_2, 2, 1).reshape((x, y*z), order='A')

		#savetxt_compact("output",sav)
		

	def test_whole(self):
		dic = DIC_NR()
		dic.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "def50.bmp", 7, [0, 0])
		print("Not running whole calculation")
		'''result = dic.calculate()
		x,y,z = result.shape
		sav = np.swapaxes(result, 2, 1).reshape((x, y*z), order='A')

		savetxt_compact("output", sav)'''

	''' no file def_500_19?
	def test_gen(self):
		#test_image_1 = np.array(Image.open(TEST_IMAGE_DIR + "def_500_19.bmp").convert('LA')) # numpy.array#
		#test_image_1 = test_image_1.astype('d')
		#print(test_image_1)

		dic = DIC_NR()
		dic.set_parameters(TEST_IMAGE_DIR + "ref_500_19.bmp", TEST_IMAGE_DIR + "def_500_19.bmp", 11, [0,0])
		dic.initial_guess()
		dic.fit_spline()
		output = dic.calculate()

		x,y,z = output.shape
		sav = np.swapaxes(output_2, 2, 1).reshape((x, y*z), order='A')

		#savetxt_compact("output",sav)
	'''

def savetxt_compact(fname, x, fmt="%.6g", delimiter=','):
	with open(f"compact_{fname}.csv", 'w+') as fh:
		for row in x:
			line = delimiter.join("0" if value == 0 else fmt % value for value in row)
			fh.write(line + '\n')

def savetxt_compact_matlab(fname, x, fmt="%.6g", delimiter=','):
	with open(f"matlab_{fname}.csv", 'w+') as fh:
		for row in x:
			line = delimiter.join("0" if value == 0 else fmt % value for value in row)
			fh.write(line + '\n')