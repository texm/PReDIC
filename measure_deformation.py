#!/bin/python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import predic as dm

def main():
	parser = argparse.ArgumentParser(description="Measure deformation between two images, given a subset size and optional initial guess.")

	parser.add_argument("ref_image", metavar="ref_image", type=str,
						help="The reference image to calculate deformations to")

	parser.add_argument("def_image", metavar="def_image", type=str,
						help="The deformed image to calculate deformations from")

	parser.add_argument("-s", "--subset_size", metavar="N", type=int,
						nargs="?", default=21, required=False,
						help="The subset size to use. Default=11")

	parser.add_argument("-i", "--initial_guess", metavar="N", type=float, 
						nargs=2, default=[0.0, 0.0], required=False,
						action="store", dest="ini_guess",
						help="""Set the initial guess to work from. 
						Defaults to [0.0, 0.0].
						Example: -i 1.0 1.0""")

	parser.add_argument("-d", "--debug", dest="debug_mode", action="store_true", 
						help="Use debug print mode.")


	parser.add_argument("-p", "--parallel", dest="parallel_mode", action="store_true", 
						help="Run in parallel *Please note that to run in parallel at larger image sizes will require a manual tweak of the file in your sitepackages/joblib/parallel.py line 475 change max_nbytes=1M to max_nbytes=50M or larger*")

	parser.add_argument("-o", "--output", dest="output_file", 
						type=str, required=False,
						help="File to write formatted csv output.")


	parser.add_argument("-v", "--visualise", dest="visualise", action="store_true",
						help="Automatically use matplotlib to visualise the output.")

	args = parser.parse_args()


	dic = dm.DIC_NR(args.debug_mode, args.parallel_mode)


	dic.set_parameters(args.ref_image, args.def_image, args.subset_size, args.ini_guess)
	results = dic.calculate()

	x,y,z = results.shape
	output = np.swapaxes(results, 2, 1).reshape((x, y*z), order="A")

	def vis_plotter(results, arr_name):
		x = results[:,:,0]
		y = results[:,:,0]
		residual = (x**2 + y**2 )**0.5

		ig, ax = plt.subplots()

		plt.subplot(1, 2, 1)
		plt.title(arr_name + " X Deformations")
		imgplot = plt.imshow(x)
		imgplot.set_cmap('YlGnBu')
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		plt.colorbar()

		plt.subplot(1, 2, 2)
		plt.title(arr_name + " Y Deformations")
		imgplot = plt.imshow(y)
		imgplot.set_cmap('YlGnBu')
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		plt.colorbar()

		plt.show()
		plt.close()

	if args.visualise:
		vis_plotter(results, args.def_image)

	if args.output_file != None:
		with open(args.output_file, 'w+') as fh:
			for row in output:
				line = ",".join("0" if n == 0 else f"{n:.6g}" for n in row)
				fh.write(line + '\n')
		print(f"Result written to {args.output_file}")
	else:
		for row in output:
			print(",".join("0" if n == 0 else f"{n:.6g}" for n in row))

if __name__ == '__main__':
	main()
