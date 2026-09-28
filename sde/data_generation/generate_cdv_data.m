clear; clc

x1s = 0.95;  x4s = -0.76095;  C = 0.1;  beta = 1.25;  gamma_ = 0.2;  b = 0.5;

m  = [1 2];
am = 8*sqrt(2)/pi * m.^2 .* (b^2 + m.^2 - 1) ./ ((4*m.^2 - 1).*(b^2 + m.^2));
bm = beta*b^2 ./ (b^2 + m.^2);
dm = 64*sqrt(2)/(15*pi) * (b^2 - m.^2 + 1) ./ (b^2 + m.^2);
gtm = gamma_*4*sqrt(2)/pi * m *  b^3    ./ ((4*m.^2 - 1).*(b^2 + m.^2));
gm  = gamma_*4*sqrt(2)/pi * m.^3 * b    ./ ((4*m.^2 - 1).*(b^2 + m.^2));
ep  = 16*sqrt(2)/(5*pi);

sig_true = 0.02*ones(6,1);

dt      = 0.005;
T       = 5000;
N       = round(T/dt);

dim      = 6;
u0       = [0.9; 0; 0; -0.6; 0; 0] + 0.01*randn(6,1);
Ttrans   = 200;
rng_seed = 10;

Fconst = [C*x1s; 0; 0; C*x4s; 0; 0];

fprintf('Analytic constraint residual delta2 + eps - delta1 = %+.4f (should be ~ -0.186)\n', ...
    dm(2) + ep - dm(1));

rhs_true = @(x) [ gtm(1)*x(3) - C*(x(1) - x1s) ;
    -(am(1)*x(1) - bm(1))*x(3) - C*x(2) - dm(1)*x(4)*x(6) ;
     (am(1)*x(1) - bm(1))*x(2) - gm(1)*x(1) - C*x(3) + dm(1)*x(4)*x(5) ;
    gtm(2)*x(6) - C*(x(4) - x4s) + ep*(x(2)*x(6) - x(3)*x(5)) ;
    -(am(2)*x(1) - bm(2))*x(6) - C*x(5) - dm(2)*x(4)*x(3) ;
     (am(2)*x(1) - bm(2))*x(5) - gm(2)*x(4) - C*x(6) + dm(2)*x(4)*x(2) ];

rng(rng_seed);
sqdt = sqrt(dt);
U = euler_maruyama(rhs_true, sig_true, u0, dt, N);
t_save = (0:N-1)*dt;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
out_file = fullfile(out_dir, 'cdv_training_data.mat');
save(out_file, 'U', 't_save', 'dt', 'T', 'N', 'dim', 'Ttrans', 'rng_seed', 'u0', 'sig_true', 'Fconst', 'x1s', 'x4s', 'C', 'beta', 'gamma_', 'b', 'm', 'am', 'bm', 'dm', 'gtm', 'gm', 'ep');
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
