import sys, os, unittest
PARENT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"

sys.path.append(PARENT_DIR)

from deformation_measurement import DeformationMeasurer 

class TestCase(unittest.TestCase):
	def setUp(self):
		self.dm = DeformationMeasurer()


	def test_import_successful(self):
		self.assertTrue(self.dm is not None)


if __name__ == '__main__':
	unittest.main()
