import sys, os, unittest

PARENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(PARENT_DIR)

TEST_IMAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/testing_images/"

from deformation_measurement import DIC_NR, C_First_Order

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline

class Test_C_First_Order(unittest.TestCase):
	#For quite a few of these tests I have set them up with subset size 11
	#In the case where the image is 40x40, this means Xmin,Ymin,Xp,Yp,Xmax & Ymax should all be 20
	def test_interpolation(self):
		#old testing, look to below functions for testing of actual code

		test_image_1 = np.array(Image.open(TEST_IMAGE_DIR + "ref50.bmp").convert('LA')) # numpy.array
		#print(test_image_1.shape)
		#print(test_image_1)
		test_image_b = test_image_1.astype('d')
		#print(test_image_b.shape)
		#print(test_image_b)

		X_size, Y_size, _tmp = test_image_b.shape
		#print(X_size, Y_size, _tmp)

		#col1 = test_image_b[48,0,0]
		#col2 = test_image_b[49,0,0]

		#test_image_c = test_image_b[:,:,0]
		#print(test_image_c)

		Y_size, X_size,tmp = test_image_b.shape

		X_defcoord = np.arange(0, X_size, dtype=int) # Maybe zero?
		Y_defcoord = np.arange(0, Y_size, dtype=int)

		interp = RectBivariateSpline(X_defcoord, Y_defcoord, test_image_b[:,:,0], kx=5, ky=5)
		#result1 = interp.ev(48,0)
		#result2 = interp.ev(48.5,0)
		#result3 = interp.ev(49,0)

		interp_res = np.empty([50,50])

		dic = DIC_NR()
		dic.set_parameters(TEST_IMAGE_DIR + "ref50.bmp", TEST_IMAGE_DIR + "def50.bmp", 11, [0, 0])
		cfo = dic.cfo
		cfo.define_deformed_subset([0, 0, 0.0, 0.0, 0.0, 0.0], 21, 20)
		'''
		print(i)
		print(j)
		print(I_matrix)
		print(J_matrix)
		print(N)
		print(I)
		print(I.shape)
		print(J)
		print(J.shape)
		print(X)
		print(Y)
		'''

		''' ???
		t = np.reshape(t, (11,11))
		t_t = np.transpose(t)
		print(t_t)
		#print(t_t)
		print(t.shape)

		for y in range(50):
			for x in range(50):
				interp_res[x][y] = interp.ev(x,y)

		interp_res = np.transpose(interp_res)
		print(interp_res[15:25, 15:25])

		#savetxt_compact("image_actual", test_image_c)
		#savetxt_compact("deform", interp_res)

		print(result1)
		print(result2)
		print(result3)

		print(col1)
		print(col2)
		'''

	def test_define_deformed_subset(self):
		#TODO: use the updated class version
		#i, j, I_matrix, J_matrix, N, I, J, X, Y = define_deformed_subset(11, 20, 20, 0, 0, 0, 0, 0, 0)
		'''
		print(i)
		print(j)
		print(I_matrix)
		print(J_matrix)
		print(N)
		print(I)
		print(I.shape)
		print(J)
		print(J.shape)
		print(X)
		print(Y)
		'''