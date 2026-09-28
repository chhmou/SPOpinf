clear; clc

J     = 6;
F     = 8;
sigma = 0.5;
s_true = sigma*ones(J,1);

dt      = 0.005;
T       = 1000;
N       = round(T/dt);

dim      = J;
u0       = F + 0.01*randn(J,1);
Ttrans   = 10;
rng_seed = 10;

Fconst = F*ones(J,1);

ip1 = circshift((1:J)',-1);
im1 = circshift((1:J)', 1);
im2 = circshift((1:J)', 2);

rhs_true = @(u) (u(ip1) - u(im2)).*u(im1) - u + F;

rng(rng_seed);
sqdt = sqrt(dt);
U = euler_maruyama(rhs_true, s_true, u0, dt, N);
t_save = (0:N-1)*dt;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
out_file = fullfile(out_dir, 'l96_training_data.mat');
save(out_file, 'U', 't_save', 'dt', 'T', 'N', 'dim', 'Ttrans', 'rng_seed', 'u0', 's_true', 'Fconst', 'J', 'F', 'sigma');
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
