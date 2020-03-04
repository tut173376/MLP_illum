% A fast and simple voxel traversal algorithm through a 3D space partition (grid)
% proposed by J. Amanatides and A. Woo (1987).

% % Test Nro. 1
% origin    = [15, 15, 15]';
% direction = [-0.3, -0.5, -0.7]';

% % Test Nro. 2
%origin    = [-8.5, -4.5, -9.5]';
%direction = [0.5, 0.5, 0.7]';

% Grid: dimensions
grid3D.nx = 10;
grid3D.ny = 10;
grid3D.nz = 10;
grid3D.minBound = [0, 0, 0]';
grid3D.maxBound = [1, 1, 1]';

origin    = [0.3, 0.5, 0.5]';
direction = [0.5, 0.5, 1.1]';

t = -1.0;
sp = [origin(1)+t*direction(1), origin(2)+t*direction(2), origin(3)+t*direction(3)]';

verbose = 1;
amanatidesWooAlgorithm(origin, direction, grid3D, verbose);
