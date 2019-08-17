%{
This file is the ownership of 

% Ghulam Mubashar Hassan 
% University of Western Australia, Perth, Australia

This file is provided to students for their MS research only. Under no
circumstance it may be distributed without owner's permission.

------------------------------------------------------------------------
|   DIGITAL IMAGE CORRELATION: IMAGES TO BE CORRELATED    |
------------------------------------------------------------------------

The following function is the one using C_First_Order function for optimzation 

%}


% Example Usage: DIC_NR_images("ref500.bmp", "def500.bmp", 7, [0 0])
function [result] = DIC_NR_images(ref_img, def_img, subsetSize, ini_guess)
% Variable ref_image takes B&W reference image
% Variable def_image takes B&W deformed image
% Variable subset_size must be provided with positive integer number
% mentioning the size of the subset to be used for correlation.
% Deformations greater than half of subset size cannot be measured
% Pts mentions the column number (x-axis) and row number (Y-axis) 
% represented by Xp and Yp later in the code respectively, which needs to 
% be the center of the subset for which deformation is required. 
% The points must not be on the edges of the image.
% q_0 is the initial guess for U and V. It must have two values only. If no
% initial guess can be provided then use [0 0]. 
% Initial guess must be in
% the range of subset size. The values on sides of images or where
% iteration process is terminated are not included in the final results.

% Global variables to be used

    global subset_size;
    global ref_image;
    global Xp;
    global Yp;
    global def_interp;
    global def_interp_x;
    global def_interp_y; 
    
    % Make sure that the subset size specified is valid (not odd at this
    % point)
    if (mod(subsetSize, 2) == 0) 
        error('Subset must be odd?');
    end
    
    % Read in the images (default directory is current working directory)
    ref_img_read = imread(ref_img);
    def_img_read = imread(def_img);
    
    % Obtain the size of the reference image
    [Y_size, X_size] = size(ref_img_read);
    
    % Convert image arrays into double arrays
    ref_image = double(ref_img_read);
    def_image = double(def_img_read);
    
    %Initalize variables
    subset_size = subsetSize;
    spline_order = 6;
    
    % termination condition for newton-raphson iteration
    Max_num_iter = 40; % maximum number of iterations
    TOL(1) = 10^(-8);  % change in correlation coefficient
    TOL(2) = 10^(-8)/2;%10^(-5)/2;% change in sum of all gradients.
%     result = zeros(length(pts(:,1)),12);
    

% condition to check that point of interest is not close to edge. Point
% must away from edge greater than half of subset adding 15 to it to have
% range of initial guess accuracy.
Xmin = round((subset_size/2) +15);
Ymin = Xmin;
Xmax = round(X_size-((subset_size/2) +15));
Ymax = round(Y_size-((subset_size/2) +15));
Xp = Xmin;
Yp = Ymin;

if ( (Xp < Xmin) || (Yp < Ymin) || (Xp > Xmax) ||  (Yp > Ymax) )
    error('Process terminated!!! First point of centre of subset is on the edge of the image. ');
end
%_____________Automatic Initial Guess______________________

% Automatic Initial Guess
q_0 = zeros(6,1);
q_0(1:2,1) = ini_guess;
% The initial guess must lie between -range to range in pixels
range = 15;%35;
u_check = (round(q_0(1)) - range):(round(q_0(1)) + range);
v_check = (round(q_0(2)) - range):(round(q_0(2)) + range);

% Define the intensities of the first reference subset
subref = ref_image(Yp-floor(subset_size/2):Yp+floor(subset_size/2), ...
                   Xp-floor(subset_size/2):Xp+floor(subset_size/2));
% Preallocate some matrix space               
sum_diff_sq = zeros(numel(u_check), numel(v_check));
% Check every value of u and v and see where the best match occurs
for iter1 = 1:numel(u_check)
    for iter2 = 1:numel(v_check)
        subdef = def_image( (Yp-floor(subset_size/2)+v_check(iter2)):(Yp+floor(subset_size/2)+v_check(iter2)), ...
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
    
end

ini_guess_formatted = '???';
if (isa(ini_guess, 'integer'))
    ini_guess_formatted = int2str(ini_guess);
elseif (ismatrix(ini_guess))
    ini_guess_formatted = mat2str(ini_guess);
end
filename = sprintf('DEFORMATION_PARAMETERS(%s, %s, %d, %s).csv', ref_img, def_img, subsetSize, ini_guess_formatted);
writematrix(DEFORMATION_PARAMETERS, filename);
result = DEFORMATION_PARAMETERS;
