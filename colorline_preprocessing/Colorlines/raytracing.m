close all;

addpath('amanatidesWooAlgorithm');

% % Test Nro. 1
origin    = [0.5, 0.5, 0.5]';
direction = [-0.3, -0.5, -0.7]';

% % Test Nro. 2
%origin    = [-8.5, -4.5, -9.5]';
%direction = [0.5, 0.5, 0.7]';

% Grid: dimensions
grid3D.nx = 10;
grid3D.ny = 10;
grid3D.nz = 10;
grid3D.minBound = [0, 0, 0]';
grid3D.maxBound = [1, 1, 1]';

verbose = 0;
ivs = amanatidesWooAlgorithm(origin, direction, grid3D, verbose);

axis([0 1 0 1 0 1])
