clear; clc; close all

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));

run_tag = char(datetime('now','Format','yyyyMMdd_HHmmss'));
out_dir = fullfile('results_l96_moi_vs_opinf_sde', ['run_' run_tag]);
if ~exist(out_dir,'dir'), mkdir(out_dir); end
results_file = fullfile(out_dir, 'log.txt');
if exist(results_file,'file'), delete(results_file); end
diary(results_file); diary on
fprintf('================================================================\n');
fprintf('  Stochastic L96 (J = 6): MOI (constrained EM) vs OpInf (LS +\n');
fprintf('  residual-calibrated noise)\n');
fprintf('  script:  l96_stochastic_moi_vs_opinf_sde.m\n');
fprintf('  run:     %s\n', run_tag);
fprintf('  outputs: %s%s\n', out_dir, filesep);
fprintf('================================================================\n\n');

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
plotTrainingData(t_save, U, Ttrans);

Us      = U(:, round(Ttrans/dt)+1:end);
z_res   = (Us(:,2:end) - Us(:,1:end-1)) / dt - Fconst;
u_train = Us(:, 1:end-1);
fprintf('Regression snapshots: %d, dim = %d  (%d coefficients)\n', ...
    size(u_train,2), dim, dim^2*(dim+1)/2 + dim^2);

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
    u_train.*sqrt(dt), z_res.*sqrt(dt), dim, J_num, theta_par, ...
    constraint_mat_H_moi, nonzero_index, tol, dt, sparsity_thresh, ...
    Gamma, Niter_em, Niter_stlsq);

[B_est, C_est, D_est] = vecToOperators(regress_theta_vec, dim);
B_est = B_est.*sqrt(dt);
D_est = Fconst;

regress_theta_vec_opinf = regress_mat_v \ regress_vec_v;
[B_ls, C_ls, D_ls] = vecToOperators(regress_theta_vec_opinf, dim);
D_ls = Fconst;

resid_ls = reshape(regress_vec_v - regress_mat_v*regress_theta_vec_opinf, dim, []);
sigma_ls = sqrt(dt * var(resid_ls, 0, 2));

sigma_hat = sqrt(diag(R_update));

[B_true, C_true, D_true] = trueL96Operators(J, F);
reportMethodComparison(B_ls, C_ls, B_est, C_est, B_true, C_true);
reportParameterError(B_est, C_est, D_est, B_true, C_true, D_true);
reportDiffusion3(sigma_hat, sigma_ls, s_true);

Tpath = 20;  Npath = round(Tpath/dt);
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

Tstat = 2000;  Nstat = round(Tstat/dt);  burn = round(20/dt);
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

writePaperSummary(B_ls, C_ls, B_est, C_est, B_true, C_true, ...
    sigma_hat, sigma_ls, s_true, dt, T, size(u_train,2), N_ens, Tstat, ...
    Utru_s, Urom_s, Uop_s);
diary off
fprintf('Command-window log written to:  %s\n', results_file);

mat_file = fullfile(out_dir, 'results.mat');
save(mat_file, 'B_true','C_true','B_est','C_est','B_ls','C_ls', ...
    'sigma_hat','sigma_ls','s_true','Fconst', ...
    'dt','T','N_ens','Tstat','run_tag');
fprintf('Key operators/metrics saved to: %s\n', mat_file);

saveAllFigures(out_dir);
fprintf('All outputs collected in:       %s%s\n', out_dir, filesep);

function [B_true, C_true, D_true] = trueL96Operators(J, F)
cyc = @(m) mod(m-1, J) + 1;
B_true = zeros(J,J,J);
for k = 1:J
    i1 = cyc(k-1);  j1 = cyc(k+1);
    B_true(min(i1,j1), max(i1,j1), k) = B_true(min(i1,j1), max(i1,j1), k) + 1;
    i2 = cyc(k-2);  j2 = cyc(k-1);
    B_true(min(i2,j2), max(i2,j2), k) = B_true(min(i2,j2), max(i2,j2), k) - 1;
end

C_true = -eye(J);
D_true = F*ones(J,1);
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
fprintf('  with 12 true nonzeros among 126 quadratic coefficients, the\n');
fprintf('  spurious-entry suppression by the constraints shows up in the\n');
fprintf('  max-abs-B row.  EM energy residual: HARD (~1e-15) vs SOFT.\n');
fprintf('===============================================================\n\n');
end

function reportDiffusion3(sigma_em, sigma_ls, sigma_true)
fprintf('==========  Diffusion recovery: EM vs residual calibration  ===\n');
fprintf('  site        MOI (EM)    OpInf (resid)      true\n');
fprintf('  ---------------------------------------------------------------\n');
for k = 1:numel(sigma_true)
    fprintf('   sigma_%d   %9.4f      %9.4f     %9.4f\n', ...
        k, sigma_em(k), sigma_ls(k), sigma_true(k));
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
J = size(U,1);
Twin = 60;
w = t >= Ttrans & t <= Ttrans + Twin;

figure('Color','w','Position',[80 60 950 420]);
imagesc(t(w), 1:J, U(:,w)); axis xy;
colormap(parula);
cb = colorbar; cb.Label.String = 'X_j';
xlabel('$t$','Interpreter','latex','FontSize',13);
ylabel('site $j$','Interpreter','latex','FontSize',13);
title(sprintf(['Training data: stochastic L96 Hovmoller ' ...
    '(J = %d, window after transient cut)'], J), ...
    'FontWeight','bold','FontSize',13);
set(gca,'FontSize',12,'Layer','top');
end

function plotPathwise3(t, Xtr, Xmo, Xop)
cTruth = [0.10 0.15 0.18];
cMOI   = [0.85 0.33 0.10];
cLS    = [0.00 0.45 0.70];

figure('Color','w','Position',[80 60 950 700]);
tl = tiledlayout(4,1,'TileSpacing','compact','Padding','compact');
for k = 1:3
    nexttile; hold on
    plot(t, Xtr(k,:), '-',  'Color',cTruth, 'LineWidth',1.2);
    plot(t, Xmo(k,:), '--', 'Color',cMOI,   'LineWidth',1.0);
    plot(t, Xop(k,:), '-.', 'Color',cLS,    'LineWidth',1.0);
    ylabel(sprintf('$X_%d$',k),'Interpreter','latex','FontSize',13);
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
title(tl,['L96 pathwise test, IDENTICAL noise: short horizon (strong ' ...
    'chaos); error-curve gap compares the methods'], ...
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
fprintf('==================  Equilibrium statistics  ===================\n');
fprintf('  var             mean         std      skewness    kurtosis\n');
fprintf('  ---------------------------------------------------------------\n');
for k = 1:3
    mt = computeMoments(Utru(k,:));
    mr = computeMoments(Urom(k,:));
    fprintf('  X_%d    truth  %10.3f %10.3f %10.3f %10.3f\n', k, mt);
    fprintf('         opinf  %10.3f %10.3f %10.3f %10.3f\n', mr);
end
mt = computeMoments(Utru(:));
mr = computeMoments(Urom(:));
fprintf('  pooled truth  %10.3f %10.3f %10.3f %10.3f\n', mt);
fprintf('         opinf  %10.3f %10.3f %10.3f %10.3f\n', mr);
fprintf('  (cyclic symmetry: per-site and pooled rows should agree; a\n');
fprintf('   model whose sites disagree has broken the symmetry)\n');
fprintf('===============================================================\n\n');
end

function plotStatistics3(Utru, Umoi, Uop, dt)
cTruth = [0.10 0.15 0.18];
cMOI   = [0.85 0.33 0.10];
cLS    = [0.00 0.45 0.70];
J = size(Utru,1);

figure('Color','w','Position',[100 60 1050 620]);
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

panels = {Utru(1,:), Umoi(1,:), Uop(1,:),  'PDF of $X_1$';
          Utru(2,:), Umoi(2,:), Uop(2,:),  'PDF of $X_2$';
          Utru(:)',  Umoi(:)',  Uop(:)',   'PDF pooled (all sites)'};
for j = 1:3
    Tk = panels{j,1};  Mk = panels{j,2};  Lk = panels{j,3};
    lo = min([Tk Mk Lk]);  hi = max([Tk Mk Lk]);
    edges = linspace(lo, hi, 80);
    ctr = (edges(1:end-1) + edges(2:end)) / 2;
    ht = histcounts(Tk, edges, 'Normalization','pdf');
    hm = histcounts(Mk, edges, 'Normalization','pdf');
    hl = histcounts(Lk, edges, 'Normalization','pdf');

    nexttile(j); hold on
    plot(ctr, ht,       'Color',cTruth, 'LineWidth',2.0);
    plot(ctr, hm, '--', 'Color',cMOI,   'LineWidth',1.8);
    plot(ctr, hl, '-.', 'Color',cLS,    'LineWidth',1.8);
    title(panels{j,4}, 'Interpreter','latex','FontWeight','bold','FontSize',11);
    xlabel('$X_j$','Interpreter','latex','FontSize',12);
    ylabel('PDF'); grid on; box on
    set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top');
    if j == 1
        legend('truth','MOI','OpInf','Box','off', ...
            'Location','north','FontSize',8);
    end
end

maxlag = round(10/dt);
lag_t  = (0:maxlag)*dt;
rTavg = zeros(1,maxlag+1);  rMavg = rTavg;  rLavg = rTavg;
for s = 1:J
    rTavg = rTavg + acfFast(Utru(s,:), maxlag)/J;
    rMavg = rMavg + acfFast(Umoi(s,:), maxlag)/J;
    rLavg = rLavg + acfFast(Uop(s,:),  maxlag)/J;
end
apanels = {acfFast(Utru(1,:),maxlag), acfFast(Umoi(1,:),maxlag), ...
           acfFast(Uop(1,:),maxlag),  'ACF of $X_1$';
           acfFast(Utru(2,:),maxlag), acfFast(Umoi(2,:),maxlag), ...
           acfFast(Uop(2,:),maxlag),  'ACF of $X_2$';
           rTavg, rMavg, rLavg,       'ACF site-averaged'};
for j = 1:3
    nexttile(3+j); hold on
    yline(0,'k-','LineWidth',0.5,'Alpha',0.3);
    plot(lag_t, apanels{j,1},       'Color',cTruth, 'LineWidth',2.0);
    plot(lag_t, apanels{j,2}, '--', 'Color',cMOI,   'LineWidth',1.8);
    plot(lag_t, apanels{j,3}, '-.', 'Color',cLS,    'LineWidth',1.8);
    title(apanels{j,4}, 'Interpreter','latex','FontWeight','bold','FontSize',11);
    xlabel('lag $\tau$','Interpreter','latex','FontSize',12);
    ylabel('ACF'); ylim([-0.55 1.02]); grid on; box on
    set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top');
end

title(tl,['L96 equilibrium: truth vs MOI SDE vs OpInf SDE ' ...
    '(cyclic symmetry: pooled PDFs, site-averaged ACFs)'], ...
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

function writePaperSummary(B_ls, C_ls, B_em, C_em, B_true, C_true, ...
    sigma_em, sigma_ls, sigma_true, dt, T, Nsnap, N_ens, Tstat, ...
    Utru, Umoi, Uop)
fro = @(X) norm(X(:));
relerr = @(Eh,Et) fro(Eh - Et) / max(fro(Et), eps);
kurt = @(a) mean(((a(:)-mean(a(:)))/std(a(:))).^4);

fprintf('\n');
fprintf('################################################################\n');
fprintf('##                    PAPER SUMMARY                           ##\n');
fprintf('################################################################\n');
fprintf('%%%% Setup (for the "numerical setup" paragraph):\n');
fprintf('   J = %d, F = 8, dt = %g, training T = %g (%d snapshots),\n', ...
    size(B_true,1), dt, T, Nsnap);
fprintf('   noise sigma = %.2g/site; statistics: %d realizations per model,\n', ...
    sigma_true(1), N_ens);
fprintf('   Tstat = %g each, distinct seed families (10xx/20xx/30xx)\n\n', Tstat);

fprintf('%%%% Drift recovery (headline; cite as rel. Frobenius errors):\n');
fprintf('   method        rel.err B      rel.err C     energy resid (truth 0)\n');
fprintf('   OpInf (LS)   %.4e   %.4e   %.4e\n', ...
    relerr(B_ls,B_true), relerr(C_ls,C_true), energyResidual(B_ls));
fprintf('   MOI  (EM)    %.4e   %.4e   %.4e\n\n', ...
    relerr(B_em,B_true), relerr(C_em,C_true), energyResidual(B_em));

fprintf('%%%% Cyclic-symmetry check (pooled kurtosis, all sites; Gauss = 3):\n');
fprintf('   truth %.3f | MOI %.3f | OpInf %.3f\n\n', ...
    kurt(Utru), kurt(Umoi), kurt(Uop));

fprintf('%%%% Diffusion (truth = %.2g per site; site spread tests symmetry):\n', ...
    sigma_true(1));
fprintf('   sigma  MOI(EM):    ');  fprintf('%.4f  ', sigma_em);  fprintf('\n');
fprintf('   sigma  OpInf(res): ');  fprintf('%.4f  ', sigma_ls);  fprintf('\n');
fprintf('   %%%% agreement between the two columns cross-validates both.\n\n');

fprintf('%%%% Suggested caption fragments:\n');
fprintf('   - Pathwise fig: "identical Wiener increments drive all three\n');
fprintf('     SDEs; the short horizon reflects L96 chaos, the error-curve\n');
fprintf('     gap the methods'' operator errors."\n');
fprintf('   - Statistics fig: "independent seeds per model, %d realizations\n', N_ens);
fprintf('     pooled; PDFs pooled over sites and ACFs site-averaged by\n');
fprintf('     cyclic symmetry."\n');
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
