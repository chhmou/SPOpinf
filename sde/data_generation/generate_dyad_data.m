clear; clc

d_u   = 0.1;
d_v   = 0.5;
gam   = 1.0;
F_u   = 0.1;
sig_u = 0.15;
sig_v = 0.5;
s_true = [sig_u; sig_v];

dt      = 0.005;
T       = 2000;
N       = round(T/dt);

u0       = [0.1; -0.1];
dim      = 2;
Ttrans   = 20;
rng_seed = 10;

Fconst = [F_u; 0];

rhs_true = @(w) [ -d_u*w(1) + gam*w(1)*w(2) + F_u ;
    -d_v*w(2) - gam*w(1)^2       ];

rng(rng_seed);
sqdt = sqrt(dt);
U = euler_maruyama(rhs_true, s_true, u0, dt, N);
t_save = (0:N-1)*dt;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
out_file = fullfile(out_dir, 'dyad_training_data.mat');
save(out_file, 'U', 't_save', 'dt', 'T', 'N', 'dim', 'Ttrans', 'rng_seed', 'u0', 's_true', 'Fconst', 'd_u', 'd_v', 'gam', 'F_u', 'sig_u', 'sig_v');
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
