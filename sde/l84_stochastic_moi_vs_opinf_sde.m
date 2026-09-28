clear; clc; close all

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));

run_tag = char(datetime('now','Format','yyyyMMdd_HHmmss'));
out_dir = fullfile('results_l84_moi_vs_opinf_sde', ['run_' run_tag]);
if ~exist(out_dir,'dir'), mkdir(out_dir); end
results_file = fullfile(out_dir, 'log.txt');
if exist(results_file,'file'), delete(results_file); end
diary(results_file); diary on
fprintf('================================================================\n');
fprintf('  Stochastic L84: MOI (constrained EM) vs OpInf (LS + residual-\n');
fprintf('  calibrated noise)\n');
fprintf('  script:  l84_stochastic_moi_vs_opinf_sde.m\n');
fprintf('  run:     %s\n', run_tag);
fprintf('  outputs: %s%s\n', out_dir, filesep);
fprintf('================================================================\n\n');

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
plotTrainingData(t_save, U, Ttrans);

Us      = U(:, round(Ttrans/dt)+1:end);
z_res   = (Us(:,2:end) - Us(:,1:end-1)) / dt - Fconst;
u_train = Us(:, 1:end-1);
fprintf('Regression snapshots: %d, dim = %d\n', size(u_train,2), dim);
fprintf('Known forcing Fconst = [%g; %g; %g] subtracted from target\n', Fconst);

nFeat = dim*(dim+1)/2 + dim;
Phi   = zeros(size(u_train,2), nFeat);
for k = 1:size(u_train,2)
    Phi(k,:) = buildFeatures(u_train(:,k))';
end
fprintf('Feature-matrix condition number: %.3e\n', cond(Phi));

nTheta = dim^2*(dim+1)/2 + dim^2;

regress_mat_v = assembleRegressionMatrix(u_train, dim);
regress_vec_v = reshape(z_res, [], 1);

[vec_index1] =  constraint_ten_iii(dim);

[vec_index31,vec_index32,vec_index33] =  constraint_ten_iij(dim);

[vec_index61,vec_index62,vec_index63,...
    vec_index64,vec_index65,vec_index66] = constraint_ten_ijk(dim);

ten2vec_index = B_index_moi(dim);

row_pos = 0;
identity_iii = 1:dim;
row_Biii = identity_iii'+row_pos;
col_Biii =  vec_index1;

row_pos = row_pos + dim ;
identity_iij= kron(ones(3,1),[1:dim*(dim-1)]');
row_Biij = identity_iij+row_pos;
col_Biij =  [vec_index31;vec_index32;vec_index33];

row_pos = row_pos + dim*(dim-1) ;
identity_ijk= kron(ones(6,1),[1:dim*(dim-1)*(dim-2)/6]');
row_Bijk = identity_ijk+row_pos;
col_Bijk =  [vec_index61;vec_index62;vec_index63;vec_index64;vec_index65;vec_index66];

constraint_row = [ row_Biii;row_Biij;row_Bijk];
constraint_col = [col_Biii;col_Biij;col_Bijk];
constraint_val = ones(size(constraint_row,1),1);

num_row_constraint = dim+ dim*(dim-1)+dim*(dim-1)*(dim-2)/6;
num_col_constraint = dim^3+dim^2;

constraint_mat_H = sparse(constraint_row,constraint_col,constraint_val,...
    num_row_constraint,num_col_constraint);

ind_vec_upper_tri = vec_index_ten_upper_tri(dim);
ind_vec_col_constraint = [ind_vec_upper_tri;[dim^3+1:num_col_constraint]' ];

constraint_mat_H_moi = constraint_mat_H(:,ind_vec_col_constraint);

[nonzero_index] = non_zero_BC(dim);

J_num = size(u_train,2);
theta_par = zeros(nTheta,1);
tol=1e-8;

sparsity_thresh = 0.1;
Gamma       = 1e-8;
Niter_em    = 100;
Niter_stlsq = 10;
[regress_theta_vec, R_update, active] = em_stochastic_solver_sparse( ...
    u_train, z_res, dim, J_num, theta_par, ...
    constraint_mat_H_moi, nonzero_index, tol, dt, sparsity_thresh, ...
    Gamma, Niter_em, Niter_stlsq);

[B_est, C_est, D_est] = vecToOperators(regress_theta_vec, dim);
D_est = Fconst;

regress_theta_vec_opinf = regress_mat_v \ regress_vec_v;
[B_ls, C_ls, D_ls] = vecToOperators(regress_theta_vec_opinf, dim);
D_ls = Fconst;

resid_ls = reshape(regress_vec_v - regress_mat_v*regress_theta_vec_opinf, dim, []);
sigma_ls = sqrt(dt * var(resid_ls, 0, 2));

sigma_hat = sqrt(diag(R_update.*dt));

[B_true, C_true, D_true] = trueL84Operators(a, b, F, G);
reportMethodComparison(B_ls, C_ls, B_est, C_est, B_true, C_true);
reportParameterError(B_est, C_est, D_est, B_true, C_true, D_true);
reportDiffusion3(sigma_hat, sigma_ls, s_true);

Tpath = 60;  Npath = round(Tpath/dt);
rng(777);
x_tr = u0;   x_mo = u0;   x_op = u0;
Xtr = zeros(dim, Npath+1);  Xmo = Xtr;  Xop = Xtr;
Xtr(:,1) = x_tr;  Xmo(:,1) = x_mo;  Xop(:,1) = x_op;
rhs_moi = @(u) squeeze(sum(sum(B_est .* (u*u'), 1), 2)) + C_est*u + D_est;
rhs_ls  = @(u) squeeze(sum(sum(B_ls  .* (u*u'), 1), 2)) + C_ls*u  + D_ls;
for n = 1:Npath
    xi = randn(dim,1);
    x_tr = x_tr + rhs_true(x_tr)*dt + s_true   .*(sqdt*xi);
    x_mo = x_mo + rhs_moi(x_mo)*dt  + sigma_hat.*(sqdt*xi);
    x_op = x_op + rhs_ls(x_op)*dt   + sigma_ls .*(sqdt*xi);
    Xtr(:,n+1) = x_tr;  Xmo(:,n+1) = x_mo;  Xop(:,n+1) = x_op;
end
plotPathwise3((0:Npath)*dt, Xtr, Xmo, Xop);

Tstat = 2000;  Nstat = round(Tstat/dt);  burn = round(50/dt);
N_ens = 3;

Utru_s = [];  Urom_s = [];  Uop_s = [];
for e = 1:N_ens
    rng(1000 + e);
    Xc = euler_maruyama(rhs_true, s_true, u0, dt, Nstat);
    Utru_s = [Utru_s, Xc(:, burn:end)];

    rng(2000 + e);
    Xm = euler_maruyama(rhs_moi, sigma_hat, u0, dt, Nstat);
    Urom_s = [Urom_s, Xm(:, burn:end)];

    rng(3000 + e);
    Xo = euler_maruyama(rhs_ls, sigma_ls, u0, dt, Nstat);
    Uop_s = [Uop_s, Xo(:, burn:end)];
end
fprintf(['Statistics: %d independent realizations per model, ' ...
    'pooled %d samples each (all seeds distinct)\n'], N_ens, size(Utru_s,2));

reportMoments(Utru_s, Urom_s);
plotStatistics3(Utru_s, Urom_s, Uop_s, dt);

writePaperSummary(B_ls, C_ls, B_est, C_est, B_true, C_true, b, ...
    sigma_hat, sigma_ls, s_true, dt, T, size(u_train,2), N_ens, Tstat);
diary off
fprintf('Command-window log written to:  %s\n', results_file);

mat_file = fullfile(out_dir, 'results.mat');
save(mat_file, 'B_true','C_true','B_est','C_est','B_ls','C_ls', ...
    'sigma_hat','sigma_ls','s_true','Fconst', ...
    'dt','T','N_ens','Tstat','run_tag');
fprintf('Key operators/metrics saved to: %s\n', mat_file);

saveAllFigures(out_dir);
fprintf('All outputs collected in:       %s%s\n', out_dir, filesep);

function [B_true, C_true, D_true] = trueL84Operators(a, b, F, G)
B_true = zeros(3,3,3);
B_true(2,2,1) = -1;
B_true(3,3,1) = -1;
B_true(1,2,2) =  1;
B_true(1,3,2) = -b;
B_true(1,2,3) =  b;
B_true(1,3,3) =  1;

C_true = [ -a,  0,  0;
    0, -1,  0;
    0,  0, -1 ];
D_true = [a*F; G; 0];
end

function [B_est, C_est, D_est] = vecToOperators(theta, dim)
col_u1_shift = dim*dim*(dim+1)/2;
col_u2_shift = dim*dim;

B_vec = theta(1:col_u1_shift);
C_vec = theta(col_u1_shift+1 : col_u1_shift+col_u2_shift);
D_vec = theta(col_u1_shift+col_u2_shift+1 : end);

[ten2vec] = B_index_moi(dim);
B_zeros = zeros(dim^3, 1);
B_zeros(ten2vec) = B_vec;

B_est = reshape(B_zeros, dim, dim, dim);
C_est = reshape(C_vec, dim, dim)';
D_est = D_vec;
end

function regress_mat_v = assembleRegressionMatrix(u_train, dim)
Npt = size(u_train, 2);

identity     = speye(dim);
col_u1_shift = dim*dim*(dim+1)/2;

[ind_vec] = kronector_upper_tri(dim);

row_c = cell(Npt,1);  col_c = cell(Npt,1);  val_c = cell(Npt,1);

for k = 1:Npt
    uk = u_train(1:dim,k)';

    kron_k         = kron(uk, uk);
    kron_k_reduced = kron_k(ind_vec);

    bloc_u1 = kron(identity, kron_k_reduced);
    bloc_u2 = kron(identity, uk);

    [row_u1, col_u1, val_u1] = find(bloc_u1);
    [row_u2, col_u2, val_u2] = find(bloc_u2);

    row_ushift = (k-1)*dim;

    row_c{k} = [row_u1 + row_ushift; row_u2 + row_ushift];
    col_c{k} = [col_u1;              col_u2 + col_u1_shift];
    val_c{k} = [val_u1;              val_u2];
end

row_u_all = vertcat(row_c{:});
col_u_all = vertcat(col_c{:});
val_u_all = vertcat(val_c{:});

nCols = dim^2*(dim+1)/2 + dim^2;
regress_mat_v = sparse(row_u_all, col_u_all, val_u_all, dim*Npt, nCols);
end

function res = energyResidual(B)
dim  = size(B,1);
coef = containers.Map('KeyType','char','ValueType','double');
for k = 1:dim
    for i = 1:dim
        for j = i:dim
            bij = B(i,j,k);
            if bij == 0, continue; end
            tri = sort([i j k]);
            key = sprintf('%d-%d-%d', tri(1), tri(2), tri(3));
            if isKey(coef,key), coef(key) = coef(key) + bij;
            else,               coef(key) = bij; end
        end
    end
end
if coef.Count == 0, res = 0; else, res = max(abs(cell2mat(values(coef)))); end
end

function reportMethodComparison(B_ls, C_ls, B_em, C_em, B_true, C_true)
fro = @(X) norm(X(:));
relerr = @(Eh,Et) fro(Eh - Et) / max(fro(Et), eps);

fprintf('=====  METHOD COMPARISON: plain OpInf (LS) vs MOI (EM)  =======\n');
fprintf('  metric                          OpInf (LS)        MOI (EM)\n');
fprintf('  ---------------------------------------------------------------\n');
fprintf('   rel. err B (quad)          %14.4e   %14.4e\n', ...
    relerr(B_ls,B_true), relerr(B_em,B_true));
fprintf('   rel. err C (linear)        %14.4e   %14.4e\n', ...
    relerr(C_ls,C_true), relerr(C_em,C_true));
fprintf('   rel. err D (const)         %14s   %14s   (known forcing)\n', ...
    'n/a', 'n/a');
fprintf('   max abs entry err, B       %14.4e   %14.4e\n', ...
    max(abs(B_ls(:)-B_true(:))), max(abs(B_em(:)-B_true(:))));
fprintf('   max abs entry err, C       %14.4e   %14.4e\n', ...
    max(abs(C_ls(:)-C_true(:))), max(abs(C_em(:)-C_true(:))));
fprintf('   energy residual of B_hat   %14.4e   %14.4e   (truth: 0)\n', ...
    energyResidual(B_ls), energyResidual(B_em));
fprintf('  ---------------------------------------------------------------\n');
fprintf('  LS is unbiased for the drift, so both columns should be small;\n');
fprintf('  the EM energy residual shows whether the constraints (iij + ijk\n');
fprintf('  both active for L84) are enforced HARD (~1e-15) or SOFT.\n');
fprintf('===============================================================\n\n');
end

function reportDiffusion3(sigma_em, sigma_ls, sigma_true)
fprintf('==========  Diffusion recovery: EM vs residual calibration  ===\n');
fprintf('  component    MOI (EM)    OpInf (resid)      true\n');
fprintf('  ---------------------------------------------------------------\n');
nm = {'sigma_x','sigma_y','sigma_z'};
for k = 1:numel(sigma_true)
    fprintf('   %-9s  %9.4f      %9.4f     %9.4f\n', ...
        nm{k}, sigma_em(k), sigma_ls(k), sigma_true(k));
end
fprintf('===============================================================\n\n');
end

function reportParameterError(B_est, C_est, D_est, B_true, C_true, D_true)
fro = @(X) norm(X(:));
relerr = @(Eh,Et) fro(Eh - Et) / max(fro(Et), eps);

errB = relerr(B_est, B_true);
errC = relerr(C_est, C_true);
errD = relerr(D_est, D_true);

theta_est  = [B_est(:);  C_est(:);  D_est(:)];
theta_true = [B_true(:); C_true(:); D_true(:)];
errAll = norm(theta_est - theta_true) / max(norm(theta_true), eps);

maxAbsB = max(abs(B_est(:) - B_true(:)));
maxAbsC = max(abs(C_est(:) - C_true(:)));
maxAbsD = max(abs(D_est(:) - D_true(:)));

tol      = 1e-8;
true_nz  = abs(theta_true) > tol;
est_nz   = abs(theta_est)  > tol;
missed   = sum( true_nz & ~est_nz);
spurious = sum(~true_nz &  est_nz);

fprintf('\n=========  MOI drift recovery error (noise-limited)  ==========\n');
fprintf('Operator      rel. Frobenius err     max abs entry err\n');
fprintf('---------------------------------------------------------------\n');
fprintf('  B (quad)    %18.4e   %18.4e\n', errB, maxAbsB);
fprintf('  C (linear)  %18.4e   %18.4e\n', errC, maxAbsC);
fprintf('  D (const)   %18.4e   %18.4e   (known, not inferred)\n', errD, maxAbsD);
fprintf('---------------------------------------------------------------\n');
fprintf('  all stacked %18.4e\n', errAll);
fprintf('---------------------------------------------------------------\n');
fprintf('  support: %d true nonzeros | %d missed | %d spurious\n', ...
    sum(true_nz), missed, spurious);
fprintf('===============================================================\n\n');
end

function plotTrainingData(t, U, Ttrans)
col = [0.85 0.33 0.10; 0.00 0.45 0.70; 0.10 0.55 0.25];
lbl = {'$x$','$y$','$z$'};
Twin = 100;
s = t <= Ttrans + Twin;

figure('Color','w','Position',[80 60 950 560]);
tl = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
for k = 1:3
    nexttile; hold on
    plot(t(s), U(k,s), 'Color', col(k,:), 'LineWidth', 0.7);
    xline(Ttrans, 'r--', 'LineWidth', 1.0);
    ylabel(lbl{k},'Interpreter','latex','FontSize',14);
    grid on; box on; set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top');
    if k == 1, title('dashed = transient cut','FontSize',10); end
    if k < 3, set(gca,'XTickLabel',[]); end
end
xlabel('$t$','Interpreter','latex','FontSize',14);
title(tl,'Training data: stochastic Lorenz-84 (window)', ...
    'FontSize',13,'FontWeight','bold');
end

function plotPathwise3(t, Xtr, Xmo, Xop)
cTruth = [0.10 0.15 0.18];
cMOI   = [0.85 0.33 0.10];
cLS    = [0.00 0.45 0.70];
lbl    = {'$x$','$y$','$z$'};

figure('Color','w','Position',[80 60 950 700]);
tl = tiledlayout(4,1,'TileSpacing','compact','Padding','compact');
for k = 1:3
    nexttile; hold on
    plot(t, Xtr(k,:), '-',  'Color',cTruth, 'LineWidth',1.2);
    plot(t, Xmo(k,:), '--', 'Color',cMOI,   'LineWidth',1.0);
    plot(t, Xop(k,:), '-.', 'Color',cLS,    'LineWidth',1.0);
    ylabel(lbl{k},'Interpreter','latex','FontSize',13);
    grid on; box on;
    set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top','XTickLabel',[]);
    if k == 1
        legend('truth SDE','MOI SDE','OpInf SDE','Box','off', ...
            'Location','northeast','FontSize',9,'NumColumns',3);
    end
end
nexttile; hold on
eM = sqrt(sum((Xtr - Xmo).^2, 1));
eL = sqrt(sum((Xtr - Xop).^2, 1));
semilogy(t, max(eM,1e-16), '--', 'Color',cMOI, 'LineWidth',1.0);
semilogy(t, max(eL,1e-16), '-.', 'Color',cLS,  'LineWidth',1.0);
set(gca,'YScale','log');
ylabel('$\|x^{\rm tru}-x^{\rm model}\|$','Interpreter','latex','FontSize',13);
xlabel('$t$','Interpreter','latex','FontSize',13);
legend('MOI','OpInf','Box','off','Location','southeast','FontSize',9);
grid on; box on; set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top');
title(tl,['L84 pathwise test, IDENTICAL noise: tracking horizon set by ' ...
    'chaos + operator error'], ...
    'FontSize',13,'FontWeight','bold');
end

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

function m = computeMoments(a)
a  = a(:);
mu = mean(a);
sg = std(a, 1);
z  = (a - mu) / sg;
m  = [mu; sg; mean(z.^3); mean(z.^4)];
end

function reportMoments(Utru, Urom)
names = {'x','y','z'};
fprintf('==================  Equilibrium statistics  ===================\n');
fprintf('  var           mean         std      skewness    kurtosis\n');
fprintf('  ---------------------------------------------------------------\n');
for k = 1:3
    mt = computeMoments(Utru(k,:));
    mr = computeMoments(Urom(k,:));
    fprintf('  %s  truth  %10.3f %10.3f %10.3f %10.3f\n', names{k}, mt);
    fprintf('     opinf  %10.3f %10.3f %10.3f %10.3f\n', mr);
end
fprintf('  (kurtosis non-excess; Gaussian = 3)\n');
fprintf('===============================================================\n\n');
end

function plotStatistics3(Utru, Umoi, Uop, dt)
cTruth = [0.10 0.15 0.18];
cMOI   = [0.85 0.33 0.10];
cLS    = [0.00 0.45 0.70];
lbl    = {'$x$','$y$','$z$'};

figure('Color','w','Position',[100 60 1050 620]);
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

for k = 1:3
    Tk = Utru(k,:);  Mk = Umoi(k,:);  Lk = Uop(k,:);
    lo = min([Tk Mk Lk]);  hi = max([Tk Mk Lk]);
    edges = linspace(lo, hi, 80);
    ctr = (edges(1:end-1) + edges(2:end)) / 2;
    ht = histcounts(Tk, edges, 'Normalization','pdf');
    hm = histcounts(Mk, edges, 'Normalization','pdf');
    hl = histcounts(Lk, edges, 'Normalization','pdf');

    nexttile(k); hold on
    plot(ctr, ht,       'Color',cTruth, 'LineWidth',2.0);
    plot(ctr, hm, '--', 'Color',cMOI,   'LineWidth',1.8);
    plot(ctr, hl, '-.', 'Color',cLS,    'LineWidth',1.8);
    title(['PDF of ' lbl{k}], 'Interpreter','latex', ...
        'FontWeight','bold','FontSize',11);
    xlabel(lbl{k},'Interpreter','latex','FontSize',12);
    ylabel('PDF'); grid on; box on
    set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top');
    if k == 1
        legend('truth','MOI','OpInf','Box','off', ...
            'Location','north','FontSize',8);
    end
end

maxlag = round(40/dt);
lag_t  = (0:maxlag)*dt;
for k = 1:3
    rT = acfFast(Utru(k,:), maxlag);
    rM = acfFast(Umoi(k,:), maxlag);
    rL = acfFast(Uop(k,:),  maxlag);

    nexttile(3+k); hold on
    yline(0,'k-','LineWidth',0.5,'Alpha',0.3);
    plot(lag_t, rT,       'Color',cTruth, 'LineWidth',2.0);
    plot(lag_t, rM, '--', 'Color',cMOI,   'LineWidth',1.8);
    plot(lag_t, rL, '-.', 'Color',cLS,    'LineWidth',1.8);
    title(['ACF of ' lbl{k}], 'Interpreter','latex', ...
        'FontWeight','bold','FontSize',11);
    xlabel('lag $\tau$','Interpreter','latex','FontSize',12);
    ylabel('ACF'); ylim([-0.55 1.02]); grid on; box on
    set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top');
end

title(tl,['L84 equilibrium: truth vs MOI SDE vs OpInf SDE ' ...
    '(row 1: marginal PDFs, row 2: ACFs)'], ...
    'FontSize',13,'FontWeight','bold');
end

function r = acfFast(x, maxlag)
x = x(:)' - mean(x);
n = numel(x);
nfft = 2^nextpow2(n + maxlag);
f = fft(x, nfft);
c = ifft(abs(f).^2);
r = real(c(1:maxlag+1)) / max(real(c(1)), eps);
end

function phi = buildFeatures(a)
a  = a(:);
r  = numel(a);
nq = r*(r+1)/2;
quad = zeros(nq,1);
idx = 0;
for i = 1:r
    for j = i:r
        idx = idx + 1;
        quad(idx) = a(i)*a(j);
    end
end
phi = [quad; a];
end

function writePaperSummary(B_ls, C_ls, B_em, C_em, B_true, C_true, b, ...
    sigma_em, sigma_ls, sigma_true, dt, T, Nsnap, N_ens, Tstat)
fro = @(X) norm(X(:));
relerr = @(Eh,Et) fro(Eh - Et) / max(fro(Et), eps);

fprintf('\n');
fprintf('################################################################\n');
fprintf('##                    PAPER SUMMARY                           ##\n');
fprintf('################################################################\n');
fprintf('%%%% Setup (for the "numerical setup" paragraph):\n');
fprintf('   dt = %g, training T = %g (%d snapshots), noise sigma = %.2g/mode\n', ...
    dt, T, Nsnap, sigma_true(1));
fprintf('   statistics: %d independent realizations per model, Tstat = %g each,\n', ...
    N_ens, Tstat);
fprintf('   distinct seed families (truth 10xx / MOI 20xx / OpInf 30xx)\n\n');

fprintf('%%%% Drift recovery (headline; cite as rel. Frobenius errors):\n');
fprintf('   method        rel.err B      rel.err C     energy resid (truth 0)\n');
fprintf('   OpInf (LS)   %.4e   %.4e   %.4e\n', ...
    relerr(B_ls,B_true), relerr(C_ls,C_true), energyResidual(B_ls));
fprintf('   MOI  (EM)    %.4e   %.4e   %.4e\n\n', ...
    relerr(B_em,B_true), relerr(C_em,C_true), energyResidual(B_em));

fprintf('%%%% Largest quadratic coefficients (truth b = %g):\n', b);
fprintf('   B(1,3,2): LS %+.4f | EM %+.4f   (truth %+.1f)\n', ...
    B_ls(1,3,2), B_em(1,3,2), -b);
fprintf('   B(1,2,3): LS %+.4f | EM %+.4f   (truth %+.1f)\n\n', ...
    B_ls(1,2,3), B_em(1,2,3), b);

fprintf('%%%% Smallest linear coefficient (truth a = 0.25; threshold 0.1):\n');
fprintf('   C(1,1): LS %+.4f | EM %+.4f   (truth -0.25)\n\n', ...
    C_ls(1,1), C_em(1,1));

fprintf('%%%% Diffusion (truth = %.2g per mode):\n', sigma_true(1));
fprintf('   sigma  MOI(EM):    ');  fprintf('%.4f  ', sigma_em);  fprintf('\n');
fprintf('   sigma  OpInf(res): ');  fprintf('%.4f  ', sigma_ls);  fprintf('\n');
fprintf('   %%%% agreement between the two columns cross-validates both.\n\n');

fprintf('%%%% Suggested caption fragments:\n');
fprintf('   - Pathwise fig: "identical Wiener increments drive all three\n');
fprintf('     SDEs; the tracking horizon reflects chaos plus operator error."\n');
fprintf('   - Statistics fig: "independent seeds per model, %d realizations\n', N_ens);
fprintf('     pooled."\n');
fprintf('################################################################\n\n');
end

function saveAllFigures(out_dir)
names = {'training_data', ...
         'pathwise_same_noise', ...
         'statistics_ensembles'};
figs = findobj('Type','figure');
[~, order] = sort([figs.Number]);
figs = figs(order);
for f = 1:numel(figs)
    if f <= numel(names), nm = names{f};
    else,                 nm = sprintf('extra%d', f);
    end
    base = fullfile(out_dir, sprintf('fig%02d_%s', f, nm));
    try
        exportgraphics(figs(f), [base '.png'], 'Resolution', 300);
    catch
        print(figs(f), [base '.png'], '-dpng', '-r300');
    end
    savefig(figs(f), [base '.fig']);
end
fprintf('Saved %d figures (.png 300dpi + .fig) to %s%s\n', ...
    numel(figs), out_dir, filesep);
end
