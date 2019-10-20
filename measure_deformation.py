#! /bin/python3

import argparse
import numpy as np
import deformation_measurement as dm

parser = argparse.ArgumentParser(description="Measure deformation between two images, given a subset size and optional initial guess.")

parser.add_argument("ref_image", metavar="ref_image", type=str,
                    help="The reference image to calculate deformations to")

parser.add_argument("def_image", metavar="def_image", type=str,
                    help="The deformed image to calculate deformations from")

parser.add_argument("-s", "--subset_size", metavar="N", type=int,
					nargs="?", default=11, required=False,
					help="The subset size to use. Default=11")

parser.add_argument("-i", "--initial_guess", metavar="N", type=float, 
					nargs=2, default=[0.0, 0.0], required=False,
					action="store", dest="ini_guess",
					help="""Set the initial guess to work from. 
					Defaults to [0.0, 0.0].
					Example: -i 1.0 1.0""")

parser.add_argument("-d", "--debug", dest="debug_mode", action="store_true", 
					help="Use debug print mode.")

parser.add_argument("-o", "--output", dest="output_file", 
					type=str, required=False,
					help="File to write formatted csv output to")

args = parser.parse_args()

dic = dm.DIC_NR()

if args.debug_mode:
	dic.enable_debug()

dic.set_parameters(args.ref_image, args.def_image, args.subset_size, args.ini_guess)
results = dic.calculate()

x,y,z = results.shape
output = np.swapaxes(results, 2, 1).reshape((x, y*z), order="A")

if args.output_file != None:
	with open(args.output_file, 'w+') as fh:
		for row in output:
			line = ",".join("0" if n == 0 else f"{n:.6g}" for n in row)
			fh.write(line + '\n')
else:
	for row in output:
		print(",".join("0" if n == 0 else f"{n:.6g}" for n in row))