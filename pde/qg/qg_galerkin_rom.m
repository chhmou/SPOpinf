clear; clc; close all

dataPath = '<path to data>';

project_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(project_dir, 'src'));
addpath(genpath(fullfile(project_dir, '..', '..', 'src')));

cfg = qg_galerkin_config(project_dir, dataPath);
par_qg = cfg.par_qg;
filenames = cfg.filenames;
r = 30;

run_tag = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
out_dir = fullfile(filenames.results_root, ['run_' run_tag]);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

log_file = fullfile(out_dir, 'log.txt');
diary(log_file); diary on
fprintf('============================================================\n');
fprintf('  2-D QG intrusive Galerkin ROM\n');
fprintf('  ROM dimension: r = %d\n', r);
fprintf('  FOM data:      %s\n', filenames.fom_data);
fprintf('  POD data:      %s\n', filenames.pod_data);
fprintf('  FOM projection:%s\n', filenames.fom_proj_data);
fprintf('  Results:       %s%s\n', out_dir, filesep);
fprintf('============================================================\n\n');

projection = load(filenames.fom_proj_data, 'w_proj');
w_proj = projection.w_proj;
assert(size(w_proj, 1) >= r, ...
    'The FOM projection contains only %d modes; requested r = %d.', ...
    size(w_proj, 1), r);

expected_saves = par_qg.n_time / par_qg.save_index + 1;
assert(size(w_proj, 2) >= expected_saves, ...
    ['The FOM projection has %d time samples, but the configured ROM ' ...
     'requires %d.'], size(w_proj, 2), expected_saves);

if ~exist(filenames.cache_root, 'dir'), mkdir(filenames.cache_root); end
operator_file = fullfile(filenames.cache_root, ...
    sprintf('qg_galerkin_operators_r%d.mat', r));

if exist(operator_file, 'file')
    cached = load(operator_file, 'qg_operator', 'operator_metadata');
    qg_operator = cached.qg_operator;
    operator_metadata = cached.operator_metadata;
    assert(cached.operator_metadata.r == r, ...
        'Cached operator dimension does not match r = %d.', r);
    fprintf('Loaded Galerkin operators from cache.\n');
else
    qg_operator = qg_rom_operator(cfg, filenames, r);
    operator_metadata = struct( ...
        'r', r, ...
        'Re', par_qg.Re, ...
        'Ro', par_qg.Ro, ...
        'N_snap', par_qg.N_snap, ...
        'pod_data', filenames.pod_data);
    save(operator_file, 'qg_operator', 'operator_metadata');
    fprintf('Saved Galerkin operators to cache.\n');
end

if ~isfield(qg_operator, 'Er')
    qg_operator.Er = qg_rom_energy_operator(par_qg, filenames, r);
    save(operator_file, 'qg_operator', 'operator_metadata');
    fprintf('Added the kinetic-energy operator to the cache.\n');
end

wr0 = w_proj(1:r, 1);

[wr_gal] = qg_rom_fun(par_qg, filenames, qg_operator, wr0, r);

t_rom = par_qg.t0 + ...
    (0:size(wr_gal, 2)-1) * par_qg.ddt * par_qg.save_index;
w_proj_compare = w_proj(1:r, 1:size(wr_gal, 2));
relative_state_error = vecnorm(wr_gal - w_proj_compare, 2, 1) ./ ...
    max(vecnorm(w_proj_compare, 2, 1), eps);
ke_gal = qg_cal_ke(wr_gal, qg_operator.Er);
ke_proj = qg_cal_ke(w_proj_compare, qg_operator.Er);

n_samples = size(wr_gal, 2);
state_difference = w_proj_compare - wr_gal;

err_galerkin = sqrt(sum(state_difference.^2, 1) / n_samples);
err_galerkin_mean = norm(mean(w_proj_compare, 2) - mean(wr_gal, 2));

w_proj_weight = sqrt(mean(w_proj_compare.^2, 2));
rmse_galerkin = sqrt(mean(state_difference.^2, 2)) ./ ...
    max(w_proj_weight, eps);

dev_truth = w_proj_compare - mean(w_proj_compare, 2);
dev_galerkin = wr_gal - mean(wr_gal, 2);
corr_denominator = sqrt(sum(dev_truth.^2, 2) .* ...
    sum(dev_galerkin.^2, 2));
corr_galerkin = sum(dev_truth .* dev_galerkin, 2) ./ ...
    max(corr_denominator, eps);

rel_entropy_galerkin = zeros(r, 1);
for mode = 1:r
    rel_entropy_galerkin(mode) = qg_cal_relative_entropy( ...
        w_proj_compare(mode, :), wr_gal(mode, :));
end

err_mean_psi = qg_cal_mean_psi_err( ...
    par_qg, filenames, wr_gal, r);
err_mean_psi_projection = qg_cal_mean_psi_err( ...
    par_qg, filenames, w_proj_compare, r);

error_metrics = struct( ...
    'relative_state_error_time', relative_state_error, ...
    'relative_trajectory_frobenius', ...
        norm(state_difference, 'fro') / ...
        max(norm(w_proj_compare, 'fro'), eps), ...
    'time_mean_coefficient_l2', err_galerkin_mean, ...
    'mode_normalized_rmse', rmse_galerkin, ...
    'mode_correlation', corr_galerkin, ...
    'mode_relative_entropy', rel_entropy_galerkin, ...
    'mean_streamfunction_l2_squared', err_mean_psi, ...
    'mean_streamfunction_l2', sqrt(err_mean_psi), ...
    'projection_mean_streamfunction_l2_squared', ...
        err_mean_psi_projection, ...
    'relative_energy_l2', norm(ke_gal - ke_proj) / ...
        max(norm(ke_proj), eps));

results_file = fullfile(out_dir, 'results.mat');
save(results_file, 'wr_gal', 'w_proj_compare', 't_rom', ...
    'relative_state_error', 'ke_gal', 'ke_proj', 'qg_operator', ...
    'err_galerkin', 'err_galerkin_mean', 'rmse_galerkin', ...
    'corr_galerkin', 'rel_entropy_galerkin', 'err_mean_psi', ...
    'err_mean_psi_projection', 'error_metrics', ...
    'par_qg', 'r', 'run_tag');

n_plot = min(4, r);
fig_coeff = figure('Color', 'w', 'Position', [100 80 1400 950]);
layout = tiledlayout(n_plot, 1, ...
    'TileSpacing', 'compact', 'Padding', 'compact');
axes_coeff = gobjects(n_plot, 1);
for mode = 1:n_plot
    axes_coeff(mode) = nexttile;
    plot(t_rom, w_proj_compare(mode, :), '-', ...
        'Color', [0.00 0.35 0.70], 'LineWidth', 1.4, ...
        'DisplayName', 'FOM projection');
    hold on
    plot(t_rom, wr_gal(mode, :), '--', ...
        'Color', [0.85 0.20 0.15], 'LineWidth', 1.4, ...
        'DisplayName', 'Galerkin ROM');
    ylabel(sprintf('w_{%d}', mode));
    grid on; box on
    if mode == 1
        legend('Location', 'best', 'Orientation', 'horizontal');
    end
    if mode < n_plot
        set(gca, 'XTickLabel', []);
    else
        xlabel('Time');
    end
end
linkaxes(axes_coeff, 'x');
title(layout, sprintf(['Resolved POD coefficients: FOM projection and ' ...
    'Galerkin ROM (r = %d)'], r));
exportgraphics(fig_coeff, fullfile(out_dir, ...
    'fig01_pod_coefficients.png'), 'Resolution', 300);
savefig(fig_coeff, fullfile(out_dir, 'fig01_pod_coefficients.fig'));

fig_error = figure('Color', 'w', 'Position', [130 130 1300 480]);
error_for_plot = relative_state_error;
error_for_plot(error_for_plot == 0) = NaN;
semilogy(t_rom, error_for_plot, 'k', 'LineWidth', 1.2);
xlabel('Time');
ylabel('||w_{G-ROM} - w_{FOM-proj}||_2 / ||w_{FOM-proj}||_2');
title(sprintf('Galerkin-ROM relative state error (r = %d)', r));
grid on; box on
positive_errors = relative_state_error(relative_state_error > 0);
if ~isempty(positive_errors)
    error_limits = 10 .^ [floor(log10(min(positive_errors))), ...
        ceil(log10(max(positive_errors)))];
    if error_limits(1) == error_limits(2)
        error_limits = error_limits .* [0.1, 10];
    end
    ylim(error_limits);
end
exportgraphics(fig_error, fullfile(out_dir, ...
    'fig02_relative_state_error.png'), 'Resolution', 300);
savefig(fig_error, fullfile(out_dir, ...
    'fig02_relative_state_error.fig'));

fig_energy = figure('Color', 'w', 'Position', [160 160 1300 520]);
semilogy(t_rom, max(ke_proj, eps), '-', ...
    'Color', [0.00 0.35 0.70], 'LineWidth', 1.6, ...
    'DisplayName', 'FOM projection');
hold on
semilogy(t_rom, max(ke_gal, eps), '--', ...
    'Color', [0.85 0.20 0.15], 'LineWidth', 1.6, ...
    'DisplayName', 'Galerkin ROM');
xlabel('Time'); ylabel('Kinetic energy');
title(sprintf('QG kinetic energy, r = %d, Re = %g, Ro = %g', ...
    r, par_qg.Re, par_qg.Ro));
legend('Location', 'best');
grid on; box on
exportgraphics(fig_energy, fullfile(out_dir, ...
    'fig03_kinetic_energy.png'), 'Resolution', 300);
savefig(fig_energy, fullfile(out_dir, 'fig03_kinetic_energy.fig'));

fig_metrics = figure('Color', 'w', 'Position', [190 190 1500 480]);
metrics_layout = tiledlayout(1, 3, ...
    'TileSpacing', 'compact', 'Padding', 'compact');
mode_index = 1:r;

nexttile
plot(mode_index, rmse_galerkin, '-o', ...
    'Color', [0.85 0.20 0.15], 'LineWidth', 1.4, 'MarkerSize', 4);
xlabel('POD mode'); ylabel('Normalized RMSE');
title('RMSE by mode'); grid on; box on; xlim([1 r]);

nexttile
plot(mode_index, corr_galerkin, '-o', ...
    'Color', [0.00 0.35 0.70], 'LineWidth', 1.4, 'MarkerSize', 4);
yline(1, ':');
xlabel('POD mode'); ylabel('Correlation');
title('Correlation by mode'); grid on; box on; xlim([1 r]); ylim([-1 1.05]);

nexttile
plot(mode_index, rel_entropy_galerkin, '-o', ...
    'Color', [0.45 0.20 0.65], 'LineWidth', 1.4, 'MarkerSize', 4);
xlabel('POD mode'); ylabel('Relative entropy');
title('KDE relative entropy by mode'); grid on; box on; xlim([1 r]);

title(metrics_layout, sprintf('Original QG Galerkin-ROM error metrics (r = %d)', r));
exportgraphics(fig_metrics, fullfile(out_dir, ...
    'fig04_modewise_error_metrics.png'), 'Resolution', 300);
savefig(fig_metrics, fullfile(out_dir, ...
    'fig04_modewise_error_metrics.fig'));

fprintf('\nIntegration samples: %d\n', size(wr_gal, 2));
fprintf('Final relative state error: %.6e\n', relative_state_error(end));
fprintf('Relative trajectory Frobenius error: %.6e\n', ...
    error_metrics.relative_trajectory_frobenius);
fprintf('Temporal-mean coefficient L2 error: %.6e\n', ...
    error_metrics.time_mean_coefficient_l2);
fprintf('Mean-streamfunction L2 error: %.6e\n', ...
    error_metrics.mean_streamfunction_l2);
fprintf('Relative kinetic-energy L2 error: %.6e\n', ...
    error_metrics.relative_energy_l2);
fprintf('Saved results to: %s\n', results_file);
fprintf('Saved diagnostics in: %s%s\n', out_dir, filesep);
diary off
