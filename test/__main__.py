import sys, os, unittest
PARENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(PARENT_DIR)

from deformation_measurement import DeformationMeasurer 

class TestCase(unittest.TestCase):
	def setUp(self):
		self.dm = DeformationMeasurer()


	def test_import_successful(self):
		self.assertTrue(self.dm is not None)


	def test_DIC_NR_images(self):
		matlab_output = [] # open csv file of matlab results
		python_output = self.dm.DIC_NR_images("ref500.bmp", "def500.bmp", 7, [0, 0])
		self.assertEqual(python_output, matlab_output)


	def test_C_First_Order(self):
		matlab_output = (0.0, 0.0, 0.0) # save matlab results as json or something?
		python_output = self.dm.C_First_Order([0, 0, 0])
		self.assertEqual(python_output, matlab_output)


if __name__ == '__main__':
	unittest.main()
