import sys, os, unittest
from timeit import default_timer as timer
PARENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(PARENT_DIR)

TEST_IMAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/testing_images/"

from deformation_measurement import DIC_NR

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline

class Test_DIC_NR(unittest.TestCase):
	def test_whole(self):
		start = timer()
		dic = DIC_NR()
		dic.set_parameters(TEST_IMAGE_DIR + "ref100_stretch.bmp", TEST_IMAGE_DIR + "def100_stretch.bmp", 21, [0, 0])
		result = dic.calculate()
		#sav = np.swapaxes(result, 2, 1).reshape((x, y*z), order='A')

		savetxt_compact("output", result)
		end = timer()
		print(end- start)

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