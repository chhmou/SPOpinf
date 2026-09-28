function W = simpson_2D_weights(nx, ny, dx, dy)

assert(mod(nx, 2) == 1 && mod(ny, 2) == 1, ...
    'Simpson quadrature requires an odd number of grid points.');

wx = 2 * ones(1, nx);
wx(2:2:end-1) = 4;
wx([1, end]) = 1;

wy = 2 * ones(ny, 1);
wy(2:2:end-1) = 4;
wy([1, end]) = 1;

W = (dy / 3) * wy * ((dx / 3) * wx);
end
