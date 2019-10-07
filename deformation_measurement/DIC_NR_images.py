import math
import numpy as np
from PIL import Image
from scipy.interpolate import splrep, PPoly
from deformation_measurement.C_First_Order import C_First_Order

def DIC_NR_images(ref_img=None,def_img=None,subsetSize=None,ini_guess=None,*args,**kwargs):
   
	# Make sure that the subset size specified is valid (not odd at this point)
	if (subsetSize % 2 == 0):
		raise ValueError("Subset size must be odd")

	global subset_size
	global ref_image
	global Xp
	global Yp
	global def_interp
	global def_interp_x
	global def_interp_y

	# Prepare for trouble (load images) (default directory is current working directory) https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
	ref_image = np.array(Image.open(ref_img).convert('LA')) # numpy.array
	def_image = np.array(Image.open(def_img).convert('LA')) # numpy.array

	# Make it double
	ref_image = ref_image.astype('d') # convert to double
	def_image = def_image.astype('d') # convert to double

	# Obtain the size of the reference image
	X_size, Y_size, _tmp = ref_image.shape

	# Initialize variables
	subset_size = subsetSize
	spline_order = 6

	# Termination condition for newton-raphson iteration
	Max_num_iter = 40 # maximum number of iterations
	TOL = [0,0]
	TOL[0] = 10**(-8) # change in correlation coeffiecient
	TOL[1] = 10**(-8)/2 # change in sum of all gradients.

	'''
	condition to check that point of interest is not close to edge. Point
	must away from edge greater than half of subset adding 15 to it to have
	range of initial guess accuracy.
	'''
	Xmin = round((subset_size/2) +15) # Might need to make it 14 cuz of arrays starting at 0
	Ymin = Xmin

	Xmax = round(X_size-((subset_size/2) +15))
	Ymax = round(Y_size-((subset_size/2) +15))
	Xp = Xmin
	Yp = Ymin

	if ( (Xp < Xmin) or (Yp < Ymin) or (Xp > Xmax) or  (Yp > Ymax) ):
		raise ValueError('Process terminated!!! First point of centre of subset is on the edge of the image. ');
	
	#_____________Automatic Initial Guess_____________

	# Automatic Initial Guess
	q_0 = np.zeros_like([], shape=(6))
	q_0[0:2] = ini_guess
	range_ = 15 # Minus 1 for array starting at zero?
	u_check = np.arange((round(q_0[0]) - range_), (round(q_0[1]) + range_)+1, dtype=int)
	v_check = np.arange((round(q_0[0]) - range_), (round(q_0[1]) + range_)+1, dtype=int)

	# Define the intensities of the first reference subset
	subref = ref_image[Yp-math.floor(subset_size/2):(Yp+math.floor(subset_size/2))+1, Xp-math.floor(subset_size/2):Xp+math.floor(subset_size/2)+1,0]
	#print(subref)
	
	# Preallocate some matrix space
	sum_diff_sq = np.zeros((u_check.size, v_check.size))
	# Check every value of u and v and see where the best match occurs
	for iter1 in range(u_check.size):
		for iter2 in range(v_check.size):
			subdef = def_image[(Yp-math.floor(subset_size/2)+v_check[iter2]):(Yp+math.floor(subset_size/2)+v_check[iter2])+1, (Xp-math.floor(subset_size/2)+u_check[iter1]):(Xp+math.floor(subset_size/2)+u_check[iter1])+1,0]

			sum_diff_sq[iter2,iter1] = sum(sum(np.square(subref-subdef)))
	#print(subdef)
	OFFSET1 = np.argmin(np.min(sum_diff_sq, axis=1)) # These offsets are +1 in MATLAB
	OFFSET2 = np.argmin(np.min(sum_diff_sq, axis=0))
	#print(OFFSET1)
	#print(OFFSET2)
	q_0[0] = u_check[OFFSET1]
	q_0[1] = u_check[OFFSET2]
	del u_check
	del v_check
	del iter1 
	del iter2 
	del subref 
	del subdef
	del sum_diff_sq
	del OFFSET1 
	del OFFSET2

	# Preallocate the matrix that holds the deformation parameter results
	DEFORMATION_PARAMETERS = np.zeros_like([], shape=(Y_size,X_size,12))

	# Set the initial guess to be the "last iteration's" solution.
	q_k = q_0[:6]


	#_______________COMPUTATIONS________________

	# Start the timer: Track the time it takes to perform the heaviest computations
	#tic????

	#__________FIT SPLINE ONTO DEFORMED SUBSET________________________
	# Obtain the size of the reference image
	Y_size, X_size,tmp = ref_image.shape
	
	# Define the deformed image's coordinates
	X_defcoord = np.arange(0, X_size, dtype=int) # Maybe zero?
	Y_defcoord = np.arange(0, Y_size, dtype=int)

	spline = splrep(X_defcoord, Y_defcoord)
	def_interp = PPoly.from_spline(spline)

	def_interp_x = def_interp([0, 1])
	def_interp_y = def_interp([1, 0])

	#_________________________________________________________________________ 
	#t_interp = toc;    # Save the amount of time it took to interpolate


	# MAIN CORRELATION LOOP -- CORRELATE THE POINTS REQUESTED

	# for i=1:length(pts(:,1))
	for yy in range(Ymin,Ymax):
		if yy > Ymin:
			q_k[:6] = DEFORMATION_PARAMETERS[yy-1,Xmin,:6]
			print(q_k)
		for xx in range(Xmin, Xmax):
			#Points for correlation and initializaing the q matrix
			Xp = xx
			Yp = yy
			#t_tmp = toc

			# __________OPTIMIZATION ROUTINE: FIND BEST FIT____________________________
			# if (itr_skip == 0)
			# Initialize some values
			n = 0
			C_last, GRAD_last, HESS = C_First_Order(q_k, globals()) # q_k was the result from last point or the user's guess
			optim_completed = False

			if np.isnan(abs(np.mean(np.mean(HESS)))):
				print(yy)
				print(xx)
				optim_completed = True
			while not optim_completed:
				# Compute the next guess and update the values
				delta_q = np.linalg.lstsq(HESS,(-GRAD_last)) # Find the difference between q_k+1 and q_k
				q_k = q_k + delta_q                             #q_k+1 = q_k + delta_q
				C, GRAD, HESS = C_First_Order(q_k, globals()) # Compute new values
				
				# Add one to the iteration counter
				n = n + 1 # Keep track of the number of iterations

				# Check to see if the values have converged according to the stopping criteria
				if n > Max_num_iter or ( abs(C-C_last) < TOL[0] and all(abs(delta_q) < TOL[1])): #needs to be tested...
					optim_completed = True
				
				C_last = C #Save the C value for comparison in the next iteration
				GRAD_last = GRAD # Save the GRAD value for comparison in the next iteration
			#_________________________________________________________________________
			#t_optim = toc - t_tmp

			#_______STORE RESULTS AND PREPARE INDICES OF NEXT SUBSET__________________
			# Store the current displacements
			DEFORMATION_PARAMETERS[yy,xx,0] = q_k[0] # displacement x
			DEFORMATION_PARAMETERS[yy,xx,1] = q_k[1] # displacement y
			DEFORMATION_PARAMETERS[yy,xx,2] = q_k[2] 
			DEFORMATION_PARAMETERS[yy,xx,3] = q_k[3] 
			DEFORMATION_PARAMETERS[yy,xx,4] = q_k[4] 
			DEFORMATION_PARAMETERS[yy,xx,5] = q_k[5] 
			DEFORMATION_PARAMETERS[yy,xx,6] = 1-C # correlation co-efficient final value
			# store points which are correlated in reference image i.e. center of subset
			DEFORMATION_PARAMETERS[yy,xx,7] = Xp 
			DEFORMATION_PARAMETERS[yy,xx,8] = Yp

			DEFORMATION_PARAMETERS[yy,xx,9] = n # number of iterations
			#DEFORMATION_PARAMETERS[yy,xx,11] = t_tmp # time of spline process
			#DEFORMATION_PARAMETERS[yy,xx,12] = t_optim # time of optimization process

		print(yy)
		print(xx)
	
	'''
	filename = 'DEFORMATION_PARAMETERS({:s}, {:s}, {:d}).csv'.format(ref_img, def_img, subsetSize)
	with open(filename, 'w') as outfile:
		for slice_2d in DEFORMATION_PARAMETERS:
			np.savetxt(outfile, slice_2d)
	outfile.close()
	'''
	return DEFORMATION_PARAMETERS

	



#DIC_NR_images("ref50.bmp", "def50.bmp", 7, [0, 0])