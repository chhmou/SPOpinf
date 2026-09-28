clear; clc

a = 0.25;  b = 4;  F = 8;  G = 1;
sigma_x = 0.1;
sigma_y = 0.1;
sigma_z = 0.1;
s_true  = [sigma_x; sigma_y; sigma_z];

dt      = 0.005;
T       = 100;
N       = round(T/dt);

u0       = [1; 1; 1];
dim      = 3;
Ttrans   = 20;
rng_seed = 10;

Fconst = [a*F; G; 0];

rhs_true = @(u) [ -u(2)^2 - u(3)^2 - a*u(1) + a*F ;
    u(1)*u(2) - b*u(1)*u(3) - u(2) + G ;
    b*u(1)*u(2) + u(1)*u(3) - u(3)     ];

rng(rng_seed);
sqdt = sqrt(dt);
U = euler_maruyama(rhs_true, s_true, u0, dt, N);
t_save = (0:N-1)*dt;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
out_file = fullfile(out_dir, 'l84_training_data.mat');
save(out_file, 'U', 't_save', 'dt', 'T', 'N', 'dim', 'Ttrans', 'rng_seed', 'u0', 's_true', 'Fconst', 'a', 'b', 'F', 'G', 'sigma_x', 'sigma_y', 'sigma_z');
fprintf('Saved %d snapshots (dim = %d, dt = %g) to %s\n', size(U, 2), size(U, 1), dt, out_file);


function U = euler_maruyama(drift, sig, u0, dt, Nsteps)
u0 = u0(:); sig = sig(:);
m  = numel(u0);
U  = zeros(m, Nsteps);
U(:,1) = u0;
for i = 2:Nsteps
    u = U(:,i-1);
    U(:,i) = u + drift(u)*dt + sig.*sqrt(dt).*randn(m,1);
end
end
