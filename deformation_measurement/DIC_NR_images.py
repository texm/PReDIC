from PIL import Image
import numpy as np
import math

def DIC_NR_images(ref_img, def_img, subsetSize, ini_guess):
	G = {}

	if subsetSize % 2 == 0:
		raise error("Subset size must be odd")

	ref_img_read = Image.open(ref_img)
	def_img_read = Image.open(def_img)

	Y_size, X_size = ref_img_read.size

	ref_image = np.asarray(ref_img_read, dtype=np.float64)
	def_image = np.asarray(def_img_read, dtype=np.float64)

	subset_size = subsetSize
	spline_order = 6

	Max_num_iter = 40
	TOL = [math.pow(10, -8), math.pow(10, -8) / 2]

	Xmin = round((subset_size / 2) + 15)
	Ymin = Xmin
	Xmax = round(X_size - (subset_size / 2) + 15)
	Ymax = round(Y_size - (subset_size / 2) + 15)

	Xp = Xmin
	Yp = Ymin

	if (Xp < Xmin) or (Yp < Ymin) or (Xp > Xmax) or (Yp > Ymax):
		raise error("Process terminated!!! First point of centre of subset is on the edge of the image.")

	#initial guess
	q_0 = np.zeros((6, 1), dtype=int)
	q_0[0][0] = ini_guess[0]
	q_0[1][0] = ini_guess[1]

	guess_range = 15 # 35
	u_check = np.arange((np.round(q_0[0]) - guess_range), (np.round(q_0[0]) + guess_range + 1))
	v_check = np.arange((np.round(q_0[1]) - guess_range), (np.round(q_0[1]) + guess_range + 1))
	
	subref_y = np.arange(Yp - np.floor(subset_size / 2), 1 + Yp + np.floor(subset_size / 2), dtype=int)
	subref_x = np.arange(Xp - np.floor(subset_size / 2), 1 + Xp + np.floor(subset_size / 2), dtype=int)

	subref = ref_image[subref_y, subref_x]
	
	sum_diff_sq = np.zeros((len(u_check), len(v_check)))

	for iter1 in range(len(u_check)):
		for iter2 in range(len(v_check)):
			subdef_y = np.arange(Yp - np.floor(subset_size / 2) + v_check[iter2], 1 + Yp + np.floor(subset_size / 2) + v_check[iter2], dtype=int)
			subdef_x = np.arange(Xp - np.floor(subset_size / 2) + u_check[iter1], 1 + Xp + np.floor(subset_size / 2) + u_check[iter1], dtype=int)
			subdef = def_image[subdef_y, subdef_x]
			
			# (subref - subdef) is erroring with subset size > 3
			sum_diff_sq[iter2, iter1] = np.sum(np.sum(np.power(subref - subdef, 2)))

	'''
% Check every value of u and v and see where the best match occurs
for iter1 = 1:numel(u_check)
	for iter2 = 1:numel(v_check)
		subdef = def_image( (Yp-floor(subset_size/2)+v_check(iter2)):(Yp+floor(subset_size/2)+v_check(iter2), ...
							(Xp-floor(subset_size/2)+u_check(iter1)):(Xp+floor(subset_size/2)+u_check(iter1)) );
		sum_diff_sq(iter2,iter1) = sum(sum( (subref - subdef).^2));
	end
end
[TMP1,OFFSET1] = min(min(sum_diff_sq,[],2));
[TMP2,OFFSET2] = min(min(sum_diff_sq,[],1));
q_0(1) = u_check(OFFSET2);
q_0(2) = v_check(OFFSET1);
clear u_check v_check iter1 iter2 subref subdef sum_diff_sq TMP1 TMP2 OFFSET1 OFFSET2;


% Preallocate the matrix that holds the deformation parameter results
DEFORMATION_PARAMETERS = zeros(Y_size,X_size,12);

% Set the initial guess to be the "last iteration's" solution.
q_k(1:6,1) = q_0(1:6,1);

%_______________COMPUTATIONS________________

% Start the timer: Track the time it takes to perform the heaviest computations
tic

%__________FIT SPLINE ONTO DEFORMED SUBSET________________________
% Obtain the size of the reference image
[Y_size, X_size] = size(ref_image);

% Define the deformed image's coordinates
X_defcoord = 1:X_size;
Y_defcoord = 1:Y_size;

% Fit the interpolating spline: g(x,y)
def_interp = spapi( {spline_order,spline_order}, {Y_defcoord, X_defcoord}, def_image(Y_defcoord,X_defcoord) );

% Find the partial derivitives of the spline: dg/dx and dg/dy
%def_interp_x = fnder(def_interp, [0,1]);
%def_interp_y = fnder(def_interp, [1,0]);

% Convert all the splines from B-form into ppform to make it
% computationally cheaper to evaluate. Also find partial derivatives of
% spline w.r.t x and y
def_interp = fn2fm(def_interp, 'pp');
def_interp_x = fnder(def_interp, [0,1]);
def_interp_y = fnder(def_interp, [1,0]);
%_________________________________________________________________________ 
t_interp = toc;    % Save the amount of time it took to interpolate


% MAIN CORRELATION LOOP -- CORRELATE THE POINTS REQUESTED

% for i=1:length(pts(:,1))
for yy = Ymin:Ymax
	if yy > Ymin 
		q_k(1:6,1) = DEFORMATION_PARAMETERS(yy-1,Xmin,1:6);
	end
	
	for xx = Xmin:Xmax
		%Points for correlation and initializaing the q matrix
		Xp = xx;
		Yp = yy;
	t_tmp = toc;

	%__________OPTIMIZATION ROUTINE: FIND BEST FIT____________________________
	
%         if (itr_skip == 0)
		
			% Initialize some values
			n = 0;
			[C_last, GRAD_last, HESS ] = C_First_Order(q_k);   % q_k was the result from last point or the user's guess
			optim_completed = false;

			if isnan(abs(mean(mean(HESS))))
				disp(yy)
				disp(xx)
				optim_completed = true;
%                 itr_skip = 1;
			end

				while optim_completed == false

					% Compute the next guess and update the values
					delta_q = HESS\(-GRAD_last);                     % Find the difference between q_k+1 and q_k
					q_k = q_k + delta_q;                             % q_k+1 = q_k + delta_q
					[C, GRAD, HESS] = C_First_Order(q_k);            % Compute new values

					% Add one to the iteration counter
					n = n + 1;                                       % Keep track of the number of iterations

					% Check to see if the values have converged according to the stopping criteria
					if n > Max_num_iter || ( abs(C-C_last) < TOL(1) && all(abs(delta_q) < TOL(2)) )
						optim_completed = true;
					end

					C_last = C;                                      % Save the C value for comparison in the next iteration
					GRAD_last = GRAD;                                % Save the GRAD value for comparison in the next iteration
				end
		
	%_________________________________________________________________________
	t_optim = toc - t_tmp;
	
	
	%_______STORE RESULTS AND PREPARE INDICES OF NEXT SUBSET__________________
	% Store the current displacements
	DEFORMATION_PARAMETERS(yy,xx,1) = q_k(1); % displacement x
	DEFORMATION_PARAMETERS(yy,xx,2) = q_k(2); % displacement y
	DEFORMATION_PARAMETERS(yy,xx,3) = q_k(3);
	DEFORMATION_PARAMETERS(yy,xx,4) = q_k(4);
	DEFORMATION_PARAMETERS(yy,xx,5) = q_k(5);
	DEFORMATION_PARAMETERS(yy,xx,6) = q_k(6);
	DEFORMATION_PARAMETERS(yy,xx,7) = 1-C; %correlation co-efficient final value
	% store points which are correlated in reference image i.e. center of subset 
	DEFORMATION_PARAMETERS(yy,xx,8) = Xp;
	DEFORMATION_PARAMETERS(yy,xx,9) = Yp;
	
	DEFORMATION_PARAMETERS(yy,xx,10) = n; % number of iterations
	DEFORMATION_PARAMETERS(yy,xx,11) = t_tmp; % time of spline process
	DEFORMATION_PARAMETERS(yy,xx,12) = t_optim; % time of optimization process
   
	 
	 
	end 
	disp(yy);disp(xx);
   '''

	result = []

	return result