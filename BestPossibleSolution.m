% The Objective of this mat file is to graphically show how the Projection
% of the vector b onto A's column space gives the best solution 
% (i.e. least square) to Ax = b
% In the case of a Rectangular or Singular b.

% Let this be a m*n matrix (m equations & n unknowns system)
A = [1 0; 5 4; 2 4]; % 3*2
A_T = A';

column1 = A_T(1,:)';
column2 = A_T(2,:)';
normal = cross(column1,column2);

%# a plane is a*x+b*y+c*z+d=0
%# [a,b,c] is the normal. Thus, we have to calculate
%# d and we're set
d = -A_T(1,:)*normal;

%# create x,y
[xx,yy]=ndgrid(0:100,0:100);

%# calculate corresponding z
z = (-normal(1)*xx - normal(2)*yy - d)/normal(3);

%# plot the surface
figure
% the points that b can take such that there exists a solution for Ax = b
surf(xx,yy,z) 