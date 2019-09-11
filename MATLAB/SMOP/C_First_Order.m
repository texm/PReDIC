
function [C, GRAD, HESS] = C_First_Order( q )

    % q is the vector of deformation variables, rename them for clarity
    u           = q(1);
    v           = q(2);
    du_dx       = q(3);
    dv_dy       = q(4);
    du_dy       = q(5);
    dv_dx       = q(6);
    
    global subset_size;
    global ref_image;
    global Xp;
    global Yp;
    global def_interp;
    global def_interp_x;
    global def_interp_y;    

    
    % i and j will define the subset points to be compared.
    i = -floor(subset_size/2) : 1 : floor(subset_size/2);
    j = -floor(subset_size/2) : 1 : floor(subset_size/2);
    
    % I_matrix and J_matrix are the grid of data points formed by vectors i and j
    [I_matrix,J_matrix] = meshgrid(i,j);
    
    % Store the number of points in the subset
    N = subset_size.*subset_size;
    
    % Reshape the I and J from grid matrices into vectors containing the (x,y) coordinates of each point
    I = reshape(I_matrix, 1,N);
    J = reshape(J_matrix, 1,N);
    
    % Since the deformed subset is not a rectangular grid this forces us to write out 
    % every x coordinate in vector X and every corresponding y in vector Y
    X = Xp + u + I + I.*du_dx + J.*du_dy;
    Y = Yp + v + J + J.*dv_dy + I.*dv_dx;
    
    
%-OBJECTIVE FUNCTION "C"---------------------------------------------------
    % f represents intensities of the discrete points in the ref subset
    f = reshape(ref_image(Yp+j, Xp+i), 1,N);
    
    % g represents the intensities of the continuous splined def sector
    g = fnval(def_interp, [Y;X]);
   
    % The following represents the double sums of C, 
    %(The summation limits are from -floor(subset_size/2) to floor(subset_size/2)
    SS_f_g = sum(sum( ((f-g).^2) ));
    SS_f_sq = sum(sum( (f.^2) ));
    
    C = SS_f_g./SS_f_sq;
%--------------------------------------------------------------------------
 
if nargout > 1
%-GRADIENT OF "C"----------------------------------------------------------  
    % Evaluate the derivitives at the points of interest
    dg_dX = fnval(def_interp_x, [Y;X]);
    dg_dY = fnval(def_interp_y, [Y;X]);
    
    % Determine the derivitives of the coordinate terms (i.e. suppose that
    % the coordinates of "f" are f(X,Y) = f(Xp+i-u+..., Yp+j-v+...)
    dX_du = 1;
    dX_dv = 0;
    dX_dudx = I;
    dX_dvdy = 0;
    dX_dudy = J;
    dX_dvdx = 0;
    
    dY_du = 0;
    dY_dv = 1;
    dY_dudx = 0;
    dY_dvdy = J;
    dY_dudy = 0;
    dY_dvdx = I;

    
    % Express the chain rule for partial derivites on "g"
    dg_du   = dg_dX.*dX_du + dg_dY.*dY_du;
    dg_dv   = dg_dX.*dX_dv + dg_dY.*dY_dv;
    dg_dudx = dg_dX.*dX_dudx + dg_dY.*dY_dudx;
    dg_dvdy = dg_dX.*dX_dvdy + dg_dY.*dY_dvdy;
    dg_dudy = dg_dX.*dX_dudy + dg_dY.*dY_dudy;
    dg_dvdx = dg_dX.*dX_dvdx + dg_dY.*dY_dvdx;
    
    % Write out each value in the gradient vector
    dC_du = sum(sum( (g-f).*(dg_du) ));
    dC_dv = sum(sum( (g-f).*(dg_dv) ));
    dC_dudx = sum(sum( (g-f).*(dg_dudx) ));
    dC_dvdy = sum(sum( (g-f).*(dg_dvdy) ));
    dC_dudy = sum(sum( (g-f).*(dg_dudy) ));
    dC_dvdx = sum(sum( (g-f).*(dg_dvdx) ));
    
    GRAD = (2/SS_f_sq).*[ dC_du, dC_dv, dC_dudx, dC_dvdy, dC_dudy, dC_dvdx ]';
%--------------------------------------------------------------------------

if nargout > 2
%-HESSIAN OF "C"-----------------------------------------------------------   

    % Write out each value in the Hessian Matrix (remember, it's symmetric,
    % so only half of the entries are need), using Knauss' approximation
    d2C_du2 = sum(sum( (dg_du).*(dg_du) ));               
    d2C_dv2 = sum(sum( (dg_dv).*(dg_dv) ));
    d2C_dudx2 = sum(sum( (dg_dudx).*(dg_dudx) ));
    d2C_dvdy2 = sum(sum( (dg_dvdy).*(dg_dvdy) ));
    d2C_dudy2 = sum(sum( (dg_dudy).*(dg_dudy) ));
    d2C_dvdx2 = sum(sum( (dg_dvdx).*(dg_dvdx) ));
    
    d2C_dudv = sum(sum( (dg_du).*(dg_dv) ));
    d2C_dududx = sum(sum( (dg_du).*(dg_dudx) ));
    d2C_dudvdy = sum(sum( (dg_du).*(dg_dvdy) ));
    d2C_dududy = sum(sum( (dg_du).*(dg_dudy) ));
    d2C_dudvdx = sum(sum( (dg_du).*(dg_dvdx) ));
    
    d2C_dvdudx = sum(sum( (dg_dv).*(dg_dudx) ));
    d2C_dvdvdy = sum(sum( (dg_dv).*(dg_dvdy) ));
    d2C_dvdudy = sum(sum( (dg_dv).*(dg_dudy) ));
    d2C_dvdvdx = sum(sum( (dg_dv).*(dg_dvdx) ));
    
    d2C_dudxdvdy = sum(sum( (dg_dudx).*(dg_dvdy) ));
    d2C_dudxdudy = sum(sum( (dg_dudx).*(dg_dudy) ));
    d2C_dudxdvdx = sum(sum( (dg_dudx).*(dg_dvdx) ));
    
    d2C_dvdydudy = sum(sum( (dg_dvdy).*(dg_dudy) ));
    d2C_dvdydvdx = sum(sum( (dg_dvdy).*(dg_dvdx) ));
    
    d2C_dudydvdx = sum(sum( (dg_dudy).*(dg_dvdx) ));
             
             
    HESS = (2/SS_f_sq).* [  d2C_du2,    d2C_dudv,   d2C_dududx,   d2C_dudvdy,   d2C_dududy,   d2C_dudvdx   ; ...
                            d2C_dudv,   d2C_dv2,    d2C_dvdudx,   d2C_dvdvdy,   d2C_dvdudy,   d2C_dvdvdx   ; ...
                            d2C_dududx, d2C_dvdudx, d2C_dudx2,    d2C_dudxdvdy, d2C_dudxdudy, d2C_dudxdvdx ; ...
                            d2C_dudvdy, d2C_dvdvdy, d2C_dudxdvdy, d2C_dvdy2,    d2C_dvdydudy, d2C_dvdydvdx ; ...
                            d2C_dududy, d2C_dvdudy, d2C_dudxdudy, d2C_dvdydudy, d2C_dudy2,    d2C_dudydvdx ; ...
                            d2C_dudvdx, d2C_dvdvdx, d2C_dudxdvdx, d2C_dvdydvdx, d2C_dudydvdx, d2C_dvdx2   ];
%--------------------------------------------------------------------------
end % if nargout > 2
end % if nargout > 1

end % function