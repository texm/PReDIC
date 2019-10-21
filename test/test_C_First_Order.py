import sys, os, unittest

PARENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(PARENT_DIR)

TEST_IMAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/testing_images/"

from deformation_measurement import DIC_NR, C_First_Order

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline

class Test_C_First_Order(unittest.TestCase):
	pass