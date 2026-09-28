clear; clc; close all

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));

run_tag = char(datetime('now','Format','yyyyMMdd_HHmmss'));
out_dir = fullfile('results_cdv_moi_vs_opinf_sde', ['run_' run_tag]);
if ~exist(out_dir,'dir'), mkdir(out_dir); end
results_file = fullfile(out_dir, 'log.txt');
if exist(results_file,'file'), delete(results_file); end
diary(results_file); diary on
fprintf('================================================================\n');
fprintf('  Stochastic CDV: MOI (pruned constraints + EM) vs OpInf (LS +\n');
fprintf('  residual-calibrated noise)\n');
fprintf('  script:  cdv_stochastic_moi_vs_opinf_sde.m\n');
fprintf('  run:     %s\n', run_tag);
fprintf('  outputs: %s%s\n', out_dir, filesep);
fprintf('================================================================\n\n');

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

[B_true, C_true, D_true] = trueCDVOperators(am, bm, dm, gtm, gm, ep, C, x1s, x4s);

theta_true_moi = thetaMOI(B_true, C_true, ten2vec_index);
viol = constraint_mat_H_moi * theta_true_moi;
drop = abs(viol) > 1e-10;
fprintf('Constraint pruning: dropping %d of %d rows violated by the truth\n', ...
    nnz(drop), size(constraint_mat_H_moi,1));
fprintf('  violated-row residuals: %s  (each = +/-(delta2+eps-delta1))\n', ...
    mat2str(full(viol(drop))', 4));
constraint_mat_H_moi = constraint_mat_H_moi(~drop, :);

[nonzero_index] = non_zero_BC(dim);

J_num = size(u_train,2);
theta_par = zeros(nTheta,1);
tol=1e-8;

sparsity_thresh = 0.0;
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

reportMethodComparison(B_ls, C_ls, B_est, C_est, B_true, C_true, ...
    constraint_mat_H_moi, ten2vec_index);
reportParameterError(B_est, C_est, D_est, B_true, C_true, D_true);
reportSmallCoeffs2(B_ls, C_ls, B_est, C_est, am, bm, dm, gtm, gm, ep, C);
reportDiffusion3(sigma_hat, sigma_ls, sig_true);

Tpath = 400;  Npath = round(Tpath/dt);
rng(777);
x_tr = u0;   x_mo = u0;   x_op = u0;
Xtr = zeros(dim, Npath+1);  Xmo = Xtr;  Xop = Xtr;
Xtr(:,1) = x_tr;  Xmo(:,1) = x_mo;  Xop(:,1) = x_op;
rhs_moi = @(x) squeeze(sum(sum(B_est .* (x*x'), 1), 2)) + C_est*x + D_est;
rhs_ls  = @(x) squeeze(sum(sum(B_ls  .* (x*x'), 1), 2)) + C_ls*x  + D_ls;
for n = 1:Npath
    xi = randn(dim,1);
    x_tr = x_tr + rhs_true(x_tr)*dt + sig_true .*(sqdt*xi);
    x_mo = x_mo + rhs_moi(x_mo)*dt  + sigma_hat.*(sqdt*xi);
    x_op = x_op + rhs_ls(x_op)*dt   + sigma_ls .*(sqdt*xi);
    Xtr(:,n+1) = x_tr;  Xmo(:,n+1) = x_mo;  Xop(:,n+1) = x_op;
end
plotPathwise3((0:Npath)*dt, Xtr, Xmo, Xop);

Tstat = 4000;  Nstat = round(Tstat/dt);  burn = round(200/dt);
N_ens = 3;

Utru_s = [];  Urom_s = [];  Uop_s = [];
for e = 1:N_ens
    rng(1000 + e);
    Xc = euler_maruyama(rhs_true, sig_true, u0, dt, Nstat);
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
plotJointRegimes3(Utru_s, Urom_s, Uop_s);

writePaperSummary(B_ls, C_ls, B_est, C_est, B_true, C_true, ...
    constraint_mat_H_moi, ten2vec_index, gtm, gm, ...
    sigma_hat, sigma_ls, sig_true, dt, T, size(u_train,2), N_ens, Tstat);
diary off
fprintf('Command-window log written to:  %s\n', results_file);

mat_file = fullfile(out_dir, 'results.mat');
save(mat_file, 'B_true','C_true','B_est','C_est','B_ls','C_ls', ...
    'sigma_hat','sigma_ls','sig_true','Fconst', ...
    'dt','T','N_ens','Tstat','run_tag');
fprintf('Key operators/metrics saved to: %s\n', mat_file);

saveAllFigures(out_dir);
fprintf('All outputs collected in:       %s%s\n', out_dir, filesep);

function [B_true, C_true, D_true] = trueCDVOperators(am, bm, dm, gtm, gm, ep, C, x1s, x4s)
B_true = zeros(6,6,6);
B_true(1,3,2) = -am(1);
B_true(4,6,2) = -dm(1);
B_true(1,2,3) =  am(1);
B_true(4,5,3) =  dm(1);
B_true(2,6,4) =  ep;
B_true(3,5,4) = -ep;
B_true(1,6,5) = -am(2);
B_true(3,4,5) = -dm(2);
B_true(1,5,6) =  am(2);
B_true(2,4,6) =  dm(2);

C_true = zeros(6,6);
C_true(1,1) = -C;  C_true(1,3) =  gtm(1);
C_true(2,2) = -C;  C_true(2,3) =  bm(1);
C_true(3,1) = -gm(1);  C_true(3,2) = -bm(1);  C_true(3,3) = -C;
C_true(4,4) = -C;  C_true(4,6) =  gtm(2);
C_true(5,5) = -C;  C_true(5,6) =  bm(2);
C_true(6,4) = -gm(2);  C_true(6,5) = -bm(2);  C_true(6,6) = -C;

D_true = [C*x1s; 0; 0; C*x4s; 0; 0];
end

function theta = thetaMOI(B, C, ten2vec_index)
Bflat = B(:);
theta = [Bflat(ten2vec_index); reshape(C', [], 1)];
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

function reportMethodComparison(B_ls, C_ls, B_em, C_em, B_true, C_true, ...
    H_kept, ten2vec_index)
fro = @(X) norm(X(:));
relerr = @(Eh,Et) fro(Eh - Et) / max(fro(Et), eps);
hres = @(B,Cm) full(max(abs(H_kept * thetaMOI(B, Cm, ten2vec_index))));
eT = energyResidual(B_true);

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
fprintf('   |H_kept * theta_hat|       %14.4e   %14.4e   (truth: 0)\n', ...
    hres(B_ls, C_ls), hres(B_em, C_em));
fprintf('   energy residual of B_hat   %14.4f   %14.4f   (TRUTH: %.4f)\n', ...
    energyResidual(B_ls), energyResidual(B_em), eT);
fprintf('  ---------------------------------------------------------------\n');
fprintf('  LS is unbiased for the drift, so both columns should be small;\n');
fprintf('  the EM column should be tighter on the H_kept metric (enforced)\n');
fprintf('  and its energy residual should sit NEAR THE TRUE 0.186 -- for\n');
fprintf('  CDV, matching the truth residual (not 0) is the correct target.\n');
fprintf('===============================================================\n\n');
end

function reportSmallCoeffs2(B_ls, C_ls, B_em, C_em, am, bm, dm, gtm, gm, ep, C)
rows = { ...
  'a1 (B 1,3,2)',   -am(1),  B_ls(1,3,2),  B_em(1,3,2);
  'd1 (B 4,5,3)',    dm(1),  B_ls(4,5,3),  B_em(4,5,3);
  'eps (B 2,6,4)',   ep,     B_ls(2,6,4),  B_em(2,6,4);
  'a2 (B 1,5,6)',    am(2),  B_ls(1,5,6),  B_em(1,5,6);
  'd2 (B 2,4,6)',    dm(2),  B_ls(2,4,6),  B_em(2,4,6);
  'C   (C 1,1)',    -C,      C_ls(1,1),    C_em(1,1);
  'b1  (C 2,3)',     bm(1),  C_ls(2,3),    C_em(2,3);
  'b2  (C 5,6)',     bm(2),  C_ls(5,6),    C_em(5,6);
  'gt1 (C 1,3)',     gtm(1), C_ls(1,3),    C_em(1,3);
  'g1  (C 3,1)',    -gm(1),  C_ls(3,1),    C_em(3,1);
  'gt2 (C 4,6)',     gtm(2), C_ls(4,6),    C_em(4,6);
  'g2  (C 6,4)',    -gm(2),  C_ls(6,4),    C_em(6,4)};
fprintf('====  Named CDV coefficients: truth vs OpInf(LS) vs MOI(EM)  ==\n');
fprintf('  coeff            truth      OpInf(LS)   rel%%      MOI(EM)   rel%%\n');
fprintf('  ---------------------------------------------------------------\n');
for r = 1:size(rows,1)
    tv = rows{r,2};  lv = rows{r,3};  ev = rows{r,4};
    fprintf('   %-12s %+10.5f  %+10.5f %6.1f  %+10.5f %6.1f\n', ...
        rows{r,1}, tv, lv, 100*abs(lv-tv)/abs(tv), ev, 100*abs(ev-tv)/abs(tv));
end
fprintf('  (largest rel. errors expected on tiny gt2, gt1 for both methods)\n');
fprintf('===============================================================\n\n');
end

function reportDiffusion3(sigma_em, sigma_ls, sigma_true)
fprintf('==========  Diffusion recovery: EM vs residual calibration  ===\n');
fprintf('  mode      MOI (EM)    OpInf (resid)      true\n');
fprintf('  ---------------------------------------------------------------\n');
for k = 1:numel(sigma_true)
    fprintf('   sigma_%d  %9.4f      %9.4f     %9.4f\n', ...
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
cZ = [0.10 0.15 0.18];  cB = [0.85 0.33 0.10];

figure('Color','w','Position',[80 60 950 480]);
tl = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile; hold on
plot(t, U(1,:), 'Color', cZ, 'LineWidth', 0.4);
xline(Ttrans, 'r--', 'transient cut', 'LineWidth', 1.0, 'FontSize', 9);
ylabel('$x_1$','Interpreter','latex','FontSize',14);
title('Zonal mode x_1: metastable regime switching (training data)', ...
    'FontWeight','bold');
grid on; box on; set(gca,'FontSize',12,'GridAlpha',0.12,'XTickLabel',[]);

nexttile; hold on
plot(t, U(4,:), 'Color', cB, 'LineWidth', 0.4);
xline(Ttrans, 'r--', 'LineWidth', 1.0);
ylabel('$x_4$','Interpreter','latex','FontSize',14);
xlabel('$t$','Interpreter','latex','FontSize',14);
grid on; box on; set(gca,'FontSize',12,'GridAlpha',0.12);

title(tl,'Training data: stochastic CDV (Crommelin 2004 parameters)', ...
    'FontSize',13,'FontWeight','bold');
end

function plotPathwise3(t, Xtr, Xmo, Xop)
cTruth = [0.10 0.15 0.18];
cMOI   = [0.85 0.33 0.10];
cLS    = [0.00 0.45 0.70];
pick   = [1 4];

figure('Color','w','Position',[80 60 950 640]);
tl = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
for r = 1:2
    k = pick(r);
    nexttile; hold on
    plot(t, Xtr(k,:), '-',  'Color',cTruth, 'LineWidth',1.2);
    plot(t, Xmo(k,:), '--', 'Color',cMOI,   'LineWidth',1.0);
    plot(t, Xop(k,:), '-.', 'Color',cLS,    'LineWidth',1.0);
    ylabel(sprintf('$x_%d$',k),'Interpreter','latex','FontSize',13);
    grid on; box on;
    set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top','XTickLabel',[]);
    if r == 1
        legend('truth SDE','MOI SDE','OpInf SDE','Box','off', ...
            'Location','best','FontSize',9,'NumColumns',3);
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
title(tl,['CDV pathwise test, IDENTICAL noise: do the models track the ' ...
    'truth through regime transitions?'], ...
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
fprintf('  var            mean         std      skewness    kurtosis\n');
fprintf('  ---------------------------------------------------------------\n');
for k = 1:6
    mt = computeMoments(Utru(k,:));
    mr = computeMoments(Urom(k,:));
    fprintf('  x%d  truth  %10.3f %10.3f %10.3f %10.3f\n', k, mt);
    fprintf('      opinf  %10.3f %10.3f %10.3f %10.3f\n', mr);
end
fprintf('  (x1 kurtosis < 3 signals bimodality -- check it is preserved)\n');
fprintf('===============================================================\n\n');
end

function plotStatistics3(Utru, Umoi, Uop, dt)
cTruth = [0.10 0.15 0.18];
cMOI   = [0.85 0.33 0.10];
cLS    = [0.00 0.45 0.70];
pick   = [1 2 4];

figure('Color','w','Position',[100 60 1050 620]);
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

for j = 1:3
    k = pick(j);
    Tk = Utru(k,:);  Mk = Umoi(k,:);  Lk = Uop(k,:);
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
    title(sprintf('$x_%d$', k), 'Interpreter','latex', ...
        'FontWeight','bold','FontSize',11);
    xlabel(sprintf('$x_%d$',k),'Interpreter','latex','FontSize',12);
    ylabel('PDF'); grid on; box on
    set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top');
    if j == 1
        legend('truth','MOI','OpInf','Box','off', ...
            'Location','north','FontSize',8);
    end
end

maxlag = round(100/dt);
lag_t  = (0:maxlag)*dt;
for j = 1:3
    k = pick(j);
    rT = acfFast(Utru(k,:), maxlag);
    rM = acfFast(Umoi(k,:), maxlag);
    rL = acfFast(Uop(k,:),  maxlag);

    nexttile(3+j); hold on
    yline(0,'k-','LineWidth',0.5,'Alpha',0.3);
    plot(lag_t, rT,       'Color',cTruth, 'LineWidth',2.0);
    plot(lag_t, rM, '--', 'Color',cMOI,   'LineWidth',1.8);
    plot(lag_t, rL, '-.', 'Color',cLS,    'LineWidth',1.8);
    title(sprintf('ACF $x_%d$', k), 'Interpreter','latex', ...
        'FontWeight','bold','FontSize',11);
    xlabel('lag $\tau$','Interpreter','latex','FontSize',12);
    ylabel('ACF'); ylim([-0.35 1.02]); grid on; box on
    set(gca,'FontSize',11,'GridAlpha',0.12,'Layer','top');
end

title(tl,['CDV equilibrium: truth vs MOI SDE vs OpInf SDE ' ...
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

function plotJointRegimes3(Utru, Umoi, Uop)
nb = 70;
x1all = [Utru(1,:) Umoi(1,:) Uop(1,:)];
x4all = [Utru(4,:) Umoi(4,:) Uop(4,:)];
e1 = linspace(min(x1all), max(x1all), nb+1);
e4 = linspace(min(x4all), max(x4all), nb+1);
c1 = (e1(1:end-1)+e1(2:end))/2;  c4 = (e4(1:end-1)+e4(2:end))/2;

hT = histcounts2(Utru(1,:), Utru(4,:), e1, e4, 'Normalization','pdf');
hM = histcounts2(Umoi(1,:), Umoi(4,:), e1, e4, 'Normalization','pdf');
hL = histcounts2(Uop(1,:),  Uop(4,:),  e1, e4, 'Normalization','pdf');
cmax = max([hT(:); hM(:); hL(:)]);

figure('Color','w','Position',[100 100 1200 380]);
tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
panels = {hT, 'Truth'; hM, 'MOI'; hL, 'OpInf'};
for p = 1:3
    nexttile;
    imagesc(c1, c4, panels{p,1}'); axis xy; caxis([0 cmax]);
    colormap(parula); colorbar;
    xlabel('$x_1$','Interpreter','latex','FontSize',13);
    ylabel('$x_4$','Interpreter','latex','FontSize',13);
    title(panels{p,2},'FontWeight','bold');
    set(gca,'FontSize',11,'Layer','top');
end
title(tl,'Joint (x_1, x_4) PDF: do both recovered SDEs keep the two regimes?', ...
    'FontSize',13,'FontWeight','bold');
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
    H_kept, ten2vec_index, gtm, gm, ...
    sigma_em, sigma_ls, sigma_true, dt, T, Nsnap, N_ens, Tstat)
fro = @(X) norm(X(:));
relerr = @(Eh,Et) fro(Eh - Et) / max(fro(Et), eps);
hres = @(B,Cm) full(max(abs(H_kept * thetaMOI(B, Cm, ten2vec_index))));

fprintf('\n');
fprintf('################################################################\n');
fprintf('##                    PAPER SUMMARY                           ##\n');
fprintf('################################################################\n');
fprintf('%%%% Setup (for the "numerical setup" paragraph):\n');
fprintf('   dt = %g, training T = %g (%d snapshots), noise sigma = %.3g/mode\n', ...
    dt, T, Nsnap, sigma_true(1));
fprintf('   constraints: energy set MINUS the 2 rows violated by CDV truth\n');
fprintf('   statistics: %d independent realizations per model, Tstat = %g each,\n', ...
    N_ens, Tstat);
fprintf('   distinct seed families (truth 10xx / MOI 20xx / OpInf 30xx)\n\n');

fprintf('%%%% Drift recovery (headline; cite as rel. Frobenius errors):\n');
fprintf('   method        rel.err B      rel.err C     |H_kept*theta|  energy resid\n');
fprintf('   OpInf (LS)   %.4e   %.4e   %.4e    %.4f\n', ...
    relerr(B_ls,B_true), relerr(C_ls,C_true), hres(B_ls,C_ls), energyResidual(B_ls));
fprintf('   MOI  (EM)    %.4e   %.4e   %.4e    %.4f\n', ...
    relerr(B_em,B_true), relerr(C_em,C_true), hres(B_em,C_em), energyResidual(B_em));
fprintf('   %%%% energy residual target for CDV is the TRUE value %.4f, not 0.\n\n', ...
    energyResidual(B_true));

fprintf('%%%% Topographic couplings (the hard, tiny coefficients):\n');
fprintf('   gt1: truth %+.5f | LS %+.5f | EM %+.5f\n', gtm(1), C_ls(1,3), C_em(1,3));
fprintf('   gt2: truth %+.5f | LS %+.5f | EM %+.5f\n', gtm(2), C_ls(4,6), C_em(4,6));
fprintf('   g1 : truth %+.5f | LS %+.5f | EM %+.5f\n', -gm(1), C_ls(3,1), C_em(3,1));
fprintf('   g2 : truth %+.5f | LS %+.5f | EM %+.5f\n\n', -gm(2), C_ls(6,4), C_em(6,4));

fprintf('%%%% Diffusion (truth = %.3g per mode):\n', sigma_true(1));
fprintf('   sigma  MOI(EM):    ');  fprintf('%.4f  ', sigma_em);  fprintf('\n');
fprintf('   sigma  OpInf(res): ');  fprintf('%.4f  ', sigma_ls);  fprintf('\n');
fprintf('   %%%% agreement between the two columns cross-validates both.\n\n');

fprintf('%%%% Suggested caption fragments:\n');
fprintf('   - Pathwise fig: "identical Wiener increments drive all three\n');
fprintf('     SDEs; tracking through regime transitions tests the total\n');
fprintf('     operator error."\n');
fprintf('   - Joint-PDF fig: "independent seeds per model, %d realizations\n', N_ens);
fprintf('     pooled; preservation of both regimes is the decisive test."\n');
fprintf('################################################################\n\n');
end

function saveAllFigures(out_dir)
names = {'training_data', ...
         'pathwise_same_noise', ...
         'statistics_ensembles', ...
         'joint_regimes'};
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
