clc; clear; close all

dataPath = '<path to data>';

add_mypath

[sbe_settings, filenames] = sbe_additive_settings(parentpath, dataPath);
PAR = sbe_settings.PAR;

r = 6;
environment_r = str2double(getenv('SBE_R'));
if isfinite(environment_r) && environment_r >= 1 && environment_r == floor(environment_r)
    r = environment_r;
end
train_first = 10;
training_final_time = 500;
evaluation_final_time = 1000;
pdf_start_time = 50;

run_tag = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
run_dir = fullfile(filenames.results_root, 'pathwise', ...
    sprintf('run_r%02d_%s', r, run_tag));
if ~exist(run_dir, 'dir'), mkdir(run_dir); end
log_file = fullfile(run_dir, 'log.txt');
diary(log_file); diary on
cleanup_diary = onCleanup(@() diary('off'));

fprintf('===============================================================\n');
fprintf(' Stochastic Burgers: G-ROM/OpInf deterministic-stochastic pairs vs CPOpInf\n');
fprintf(' r = %d\n', r);
fprintf(' training interval ends at T = %g\n', training_final_time);
fprintf(' evaluation interval ends at T = %g\n', evaluation_final_time);
fprintf(' legacy data: %s\n', filenames.input_folder);
fprintf(' output:      %s\n', run_dir);
fprintf('===============================================================\n\n');

snapshot_spacing = PAR.dt * PAR.t_skip;
projected_data = matfile(filenames.fom_proj_data);
noise_data = matfile(filenames.dW_data);
projected_size = size(projected_data, 'fom_proj');
noise_size = size(noise_data, 'dW');

requested_snapshots = floor(evaluation_final_time / snapshot_spacing) + 1;
noise_limited_snapshots = floor(noise_size(2) / PAR.t_skip) + 1;
num_input_snapshots = min([requested_snapshots, projected_size(2), ...
    noise_limited_snapshots]);
fom_proj = projected_data.fom_proj(:, 1:num_input_snapshots);

PAR_run = PAR;
PAR_run.N = (num_input_snapshots - 1) * PAR.t_skip + 1;
dW = noise_data.dW(:, 1:PAR_run.N-1);

[A_grom, B_grom] = burgers_sine_nonlinear(PAR, r);
residual = burgers_residue_gen(PAR_run, dW, fom_proj, ...
    A_grom, B_grom, r);

train_last_max = floor(training_final_time / snapshot_spacing) + 1;
train_last = min([train_last_max, size(residual, 2), size(fom_proj, 2)]);
if train_last < train_first
    error('run_sbe_pathwise_comparison:TrainingWindow', ...
        'Training window [%d,%d] is invalid.', train_first, train_last);
end

training_states = fom_proj(1:r, train_first:train_last);
closure_targets = residual(1:r, train_first:train_last);

[B_opinf, C_opinf, ~, diag_opinf, sigma_opinf] = ...
    fitPlainPolynomialOperators(training_states, closure_targets, ...
    snapshot_spacing);

[B_cpopinf, C_cpopinf, ~, diag_cpopinf, sigma_cpopinf] = ...
    fitSparsePolynomialOperators(training_states, closure_targets, ...
    snapshot_spacing);

fprintf('Ordinary OpInf residual norm: %.6e\n', diag_opinf.relativeResidual);
fprintf('CPOpInf active terms: %d / %d\n', ...
    nnz(diag_cpopinf.activeSupport), numel(diag_cpopinf.activeSupport));
fprintf('CPOpInf constraint residual: %.6e\n', ...
    diag_cpopinf.constraintResidual);

shared_noise_file = filenames.dW_data;
fprintf('Shared FOM noise file: %s\n', shared_noise_file);
if r <= size(dW, 1)
    dW_rom = dW(1:r, :);
else
    dW_rom = zeros(r, size(dW, 2));
    dW_rom(1:size(dW, 1), :) = dW;
end

sigma_zero = zeros(r, 1);
sigma_grom = zeros(r, 1);
num_forced = min(r, numel(PAR.sigmas));
sigma_grom(1:num_forced) = PAR.sigmas(1:num_forced);

A_opinf = A_grom - C_opinf ./ PAR.nu;
B_opinf_total = B_grom - B_opinf;
A_cpopinf = A_grom - C_cpopinf ./ PAR.nu;
B_cpopinf_total = B_grom - B_cpopinf;

u_grom = burgers_galerkin_solver_readable( ...
    PAR_run, dW_rom, sigma_zero, A_grom, B_grom, r, 'G-ROM');
u_grom_sigma = burgers_galerkin_solver_readable( ...
    PAR_run, dW_rom, sigma_grom, A_grom, B_grom, r, 'G-ROM (sigma)');
u_opinf = burgers_galerkin_solver_readable( ...
    PAR_run, dW_rom, sigma_zero, A_opinf, B_opinf_total, r, 'OpInf');
u_opinf_sigma = burgers_galerkin_solver_readable( ...
    PAR_run, dW_rom, sigma_opinf, A_opinf, B_opinf_total, r, 'OpInf (sigma)');
u_cpopinf = burgers_galerkin_solver_readable( ...
    PAR_run, dW_rom, sigma_cpopinf, A_cpopinf, B_cpopinf_total, r, 'CPOpInf');

model_name = {'G-ROM'; 'G-ROM (sigma)'; 'OpInf'; ...
    'OpInf (sigma)'; 'CPOpInf'};
plot_name = {'G-ROM', 'G-ROM (\sigma)', 'OpInf', ...
    'OpInf (\sigma)', 'CPOpInf'};
model_states = {u_grom; u_grom_sigma; u_opinf; u_opinf_sigma; u_cpopinf};
num_models = numel(model_states);
num_snapshots = size(fom_proj, 2);
for model_index = 1:num_models
    num_snapshots = min(num_snapshots, size(model_states{model_index}, 2));
end
fom_modes = fom_proj(1:r, 1:num_snapshots);
for model_index = 1:num_models
    model_states{model_index} = model_states{model_index}(:, 1:num_snapshots);
end
u_grom = model_states{1};
u_grom_sigma = model_states{2};
u_opinf = model_states{3};
u_opinf_sigma = model_states{4};
u_cpopinf = model_states{5};

time = (0:num_snapshots-1) * snapshot_spacing;
reference_bound = max(abs(fom_modes(:)));
instability_limit = max(5 * reference_bound, 0.5);
stable_through_t = zeros(num_models, 1);
last_stable = zeros(num_models, 1);
for model_index = 1:num_models
    [stable_through_t(model_index), last_stable(model_index)] = ...
        stability_horizon(model_states{model_index}, time, instability_limit);
end
common_last = max(1, min(last_stable));
common_index = 1:common_last;
pdf_index = find(time >= pdf_start_time & time <= evaluation_final_time);
if numel(pdf_index) < 20
    error('run_sbe_pathwise_comparison:PdfWindow', ...
        'The requested PDF window contains fewer than 20 snapshots.');
end

full_l2_errors = cell(num_models, 1);
common_l2_errors = cell(num_models, 1);
absolute_l2 = zeros(num_models, 1);
relative_l2 = zeros(num_models, 1);
full_horizon_relative_l2 = zeros(num_models, 1);
mode_absolute_l2 = zeros(num_models, r);
mode_relative_l2 = zeros(num_models, r);
for model_index = 1:num_models
    full_l2_errors{model_index} = calculate_l2_metrics( ...
        model_states{model_index}, fom_modes, snapshot_spacing);
    common_l2_errors{model_index} = calculate_l2_metrics( ...
        model_states{model_index}(:, common_index), ...
        fom_modes(:, common_index), snapshot_spacing);
    absolute_l2(model_index) = common_l2_errors{model_index}.absolute;
    relative_l2(model_index) = common_l2_errors{model_index}.relative;
    full_horizon_relative_l2(model_index) = ...
        full_l2_errors{model_index}.relative;
    mode_absolute_l2(model_index, :) = ...
        common_l2_errors{model_index}.modeAbsolute;
    mode_relative_l2(model_index, :) = ...
        common_l2_errors{model_index}.modeRelative;
end
l2_errors = struct('fullHorizon', {full_l2_errors}, ...
    'commonInterval', {common_l2_errors}, ...
    'commonFinalTime', time(common_last), ...
    'instabilityLimit', instability_limit);

error_table = table(model_name, absolute_l2, relative_l2, ...
    full_horizon_relative_l2, stable_through_t, ...
    'VariableNames', {'Model', 'AbsoluteL2', 'RelativeL2', ...
    'FullHorizonRelativeL2', 'StableThroughT'});
for mode_index = 1:r
    error_table.(sprintf('Mode%dAbsoluteL2', mode_index)) = ...
        mode_absolute_l2(:, mode_index);
    error_table.(sprintf('Mode%dRelativeL2', mode_index)) = ...
        mode_relative_l2(:, mode_index);
end
writetable(error_table, fullfile(run_dir, 'l2_errors.csv'));

pdf_model_name = [{'Reference'}; model_name];
pdf_finite_samples = zeros(num_models + 1, 1);
pdf_last_finite_t = zeros(num_models + 1, 1);
pdf_finite_samples(1) = numel(pdf_index);
pdf_last_finite_t(1) = time(pdf_index(end));
for model_index = 1:num_models
    pdf_finite_samples(model_index + 1) = ...
        count_finite_snapshots(model_states{model_index}, pdf_index);
    pdf_last_finite_t(model_index + 1) = ...
        last_finite_time(model_states{model_index}, time, pdf_index);
end
pdf_sample_table = table(pdf_model_name, pdf_finite_samples, ...
    pdf_last_finite_t, 'VariableNames', ...
    {'Model', 'FiniteSamples', 'LastFiniteT'});
writetable(pdf_sample_table, fullfile(run_dir, 'pdf_sample_counts.csv'));

fprintf('\nL2 errors against the projected FOM reference\n');
fprintf(' Common comparison interval: [0, %.6g]\n', time(common_last));
fprintf(' Boundedness threshold: %.6g\n', instability_limit);
disp(error_table(:, {'Model', 'RelativeL2', ...
    'FullHorizonRelativeL2', 'StableThroughT'}));
fprintf('\nPDF sampling window: [%.6g, %.6g]\n', ...
    time(pdf_index(1)), time(pdf_index(end)));
disp(pdf_sample_table);

reference_color = [0.10 0.10 0.10];
model_colors = [0.85 0.325 0.098; 0.85 0.325 0.098; ...
    0.494 0.184 0.556; 0.494 0.184 0.556; 0 0.447 0.741];
line_styles = {'-', '-.', '-', ':', '--'};
stride = max(1, floor(num_snapshots / 2500));
plot_index = 1:stride:num_snapshots;
plot_states = model_states;
for model_index = 1:num_models
    plot_states{model_index}(:, last_stable(model_index)+1:end) = NaN;
end

time_series_height = max(600, 225 * r + 80);
ft = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [50 50 1500 time_series_height]);
tiledlayout(r, 1, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
for mode_index = 1:r
    ax = nexttile;
    plot(time(plot_index), fom_modes(mode_index, plot_index), '-', ...
        'Color', reference_color, 'LineWidth', 1.15); hold on
    for model_index = 1:num_models
        plot_if_finite(time(plot_index), ...
            plot_states{model_index}(mode_index, plot_index), ...
            line_styles{model_index}, model_colors(model_index, :));
    end
    xlim([time(1), time(end)]);
    ylim(stable_plot_limits(fom_modes(mode_index, :), ...
        u_grom(mode_index, :), u_grom_sigma(mode_index, :), ...
        u_cpopinf(mode_index, :)));
    grid on; box on
    title(sprintf('Mode %d', mode_index), 'FontWeight', 'normal');
    ylabel(sprintf('$a_%d$', mode_index), 'Interpreter', 'latex');
    if mode_index == r, xlabel('$t$', 'Interpreter', 'latex'); end
    set(ax, 'FontSize', 10, 'LineWidth', 0.8);
    if mode_index == 1
        legend([{'Reference'}, plot_name], 'Location', 'best', ...
            'Box', 'off', 'NumColumns', 2);
    end
end
exportgraphics(ft, fullfile(run_dir, 'time_series_all_modes.png'), ...
    'Resolution', 220);
savefig(ft, fullfile(run_dir, 'time_series_all_modes.fig'));
close(ft);

pdf_available = stable_through_t >= time(pdf_index(end));
pdf_panel_size = 520;
pdf_width = max(1000, pdf_panel_size * r);
fp = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [50 50 pdf_width 610]);
tiledlayout(1, r, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
for mode_index = 1:r
    ax = nexttile;
    pdf_support = {fom_modes(mode_index, pdf_index)};
    for model_index = 1:num_models
        if pdf_available(model_index)
            pdf_support{end+1} = ...
                model_states{model_index}(mode_index, pdf_index);
        end
    end
    [pdf_grid, pdf_bandwidth] = build_pdf_grid(pdf_support{:});
    plot_density(fom_modes(mode_index, pdf_index), '-', ...
        reference_color, pdf_grid, pdf_bandwidth); hold on
    for model_index = 1:num_models
        if pdf_available(model_index)
            plot_density(model_states{model_index}(mode_index, pdf_index), ...
                line_styles{model_index}, model_colors(model_index, :), ...
                pdf_grid, pdf_bandwidth);
        else
            plot(nan, nan, line_styles{model_index}, ...
                'Color', model_colors(model_index, :), 'LineWidth', 1.1);
        end
    end
    xlim([pdf_grid(1), pdf_grid(end)]);
    pbaspect(ax, [1 1 1]);
    grid on; box on
    title(sprintf('Mode %d', mode_index), 'FontWeight', 'normal');
    xlabel(sprintf('$a_%d$', mode_index), 'Interpreter', 'latex');
    ylabel('density');
    set(ax, 'FontSize', 10, 'LineWidth', 0.8);
    if mode_index == 1
        legend([{'Reference'}, plot_name], 'Location', 'best', ...
            'Box', 'off', 'NumColumns', 2);
    end
end
exportgraphics(fp, fullfile(run_dir, 'pdf_long_time_all_modes.png'), ...
    'Resolution', 220);
savefig(fp, fullfile(run_dir, 'pdf_long_time_all_modes.fig'));
close(fp);

stable_model_index = find(pdf_available);
spatiotemporal_states = [{fom_modes}; model_states(stable_model_index)];
spatiotemporal_model_name = [{'Reference'}; model_name(stable_model_index)];
spatiotemporal_plot_name = [{'Reference'}; plot_name(stable_model_index).'];
num_spatiotemporal_models = numel(spatiotemporal_states);

spatial_basis = PAR.eigens(:, 1:r);
if isfield(PAR, 'dis_x') && numel(PAR.dis_x) == size(spatial_basis, 1)
    spatial_coordinate = PAR.dis_x(:);
else
    spatial_coordinate = linspace(0, PAR.L, size(spatial_basis, 1)).';
end
spatiotemporal_stride = max(1, ceil(num_snapshots / 2500));
spatiotemporal_index = unique([1:spatiotemporal_stride:num_snapshots, ...
    num_snapshots]);
spatiotemporal_fields = cell(num_spatiotemporal_models, 1);
for model_index = 1:num_spatiotemporal_models
    spatiotemporal_fields{model_index} = spatial_basis * ...
        spatiotemporal_states{model_index}(:, spatiotemporal_index);
end
spatiotemporal_color_limit = max( ...
    max(abs(spatiotemporal_fields{1}), [], 'all'), eps);

figure_height = max(700, 235 * num_spatiotemporal_models + 150);
fs = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [50 50 1700 figure_height]);
tiledlayout(num_spatiotemporal_models, 1, ...
    'TileSpacing', 'compact', 'Padding', 'compact');
for model_index = 1:num_spatiotemporal_models
    ax = nexttile;
    imagesc(time(spatiotemporal_index), spatial_coordinate, ...
        spatiotemporal_fields{model_index});
    axis xy
    xlim([time(1), time(end)]);
    ylim([0, PAR.L]);
    clim([-spatiotemporal_color_limit, spatiotemporal_color_limit]);
    yticks([0, pi, 2*pi]);
    yticklabels({'0', '\pi', '2\pi'});
    ylabel(spatiotemporal_plot_name{model_index}, ...
        'Interpreter', 'tex', 'FontWeight', 'normal');
    if model_index < num_spatiotemporal_models
        xticklabels([]);
    else
        xlabel('$t$', 'Interpreter', 'latex');
    end
    set(ax, 'FontSize', 10, 'LineWidth', 0.8);
end
colormap(fs, turbo(256));
colorbar_handle = colorbar;
colorbar_handle.Layout.Tile = 'south';
colorbar_handle.Label.String = 'u(x,t)';
exportgraphics(fs, fullfile(run_dir, ...
    'spatiotemporal_stable_models.png'), 'Resolution', 220);
savefig(fs, fullfile(run_dir, 'spatiotemporal_stable_models.fig'));
close(fs);

save(fullfile(run_dir, 'results.mat'), 'PAR', 'PAR_run', 'r', ...
    'training_final_time', 'evaluation_final_time', 'pdf_start_time', ...
    'train_first', 'train_last', 'snapshot_spacing', 'A_grom', 'B_grom', ...
    'B_opinf', 'C_opinf', 'sigma_opinf', 'diag_opinf', ...
    'B_cpopinf', 'C_cpopinf', 'sigma_cpopinf', 'diag_cpopinf', ...
    'sigma_zero', 'sigma_grom', 'shared_noise_file', 'l2_errors', 'error_table', ...
    'pdf_sample_table', 'model_name', 'time', 'fom_modes', ...
    'spatiotemporal_model_name', 'spatiotemporal_stride', ...
    'spatial_coordinate', 'spatiotemporal_color_limit', ...
    'u_grom', 'u_grom_sigma', 'u_opinf', 'u_opinf_sigma', ...
    'u_cpopinf', '-v7.3');

fprintf('\nSaved results and diagnostics in:\n %s\n', run_dir);

function metrics = calculate_l2_metrics(model_state, truth_state, dt)
state_dim = size(truth_state, 1);
if any(~isfinite(model_state(:))) || any(~isfinite(truth_state(:)))
    metrics = struct('absolute', Inf, 'relative', Inf, ...
        'modeAbsolute', Inf(1, state_dim), ...
        'modeRelative', Inf(1, state_dim));
    return
end
delta = model_state - truth_state;
metrics.absolute = sqrt(dt * sum(delta(:).^2));
metrics.relative = norm(delta, 'fro') / max(norm(truth_state, 'fro'), eps);
metrics.modeAbsolute = sqrt(dt * sum(delta.^2, 2)).';
truth_mode_norm = sqrt(dt * sum(truth_state.^2, 2)).';
metrics.modeRelative = metrics.modeAbsolute ./ max(truth_mode_norm, eps);
end

function [stable_through, last_stable] = stability_horizon(model_state, time, limit)
bounded = all(isfinite(model_state), 1) & ...
    max(abs(model_state), [], 1) <= limit;
first_unstable = find(~bounded, 1, 'first');
if isempty(first_unstable)
    last_stable = numel(time);
else
    last_stable = max(1, first_unstable - 1);
end
stable_through = time(last_stable);
end

function plot_if_finite(x, y, line_style, color)
mask = isfinite(x) & isfinite(y);
if any(mask), plot(x(mask), y(mask), line_style, 'Color', color, 'LineWidth', 1.1); end
end

function sample_count = count_finite_snapshots(model_state, index)
sample_count = nnz(all(isfinite(model_state(:, index)), 1));
end

function value = last_finite_time(model_state, time, index)
finite_mask = all(isfinite(model_state(:, index)), 1);
if any(finite_mask)
    finite_index = index(finite_mask);
    value = time(finite_index(end));
else
    value = NaN;
end
end

function [grid, bandwidth] = build_pdf_grid(varargin)
samples = [];
for input_index = 1:nargin
    samples = [samples; varargin{input_index}(:)];
end
samples = samples(isfinite(samples));
sample_min = min(samples);
sample_max = max(samples);
sample_range = sample_max - sample_min;
bandwidth = max(0.05 * sample_range, sqrt(eps));
padding = max(0.05 * sample_range, eps);
grid = linspace(sample_min - padding, sample_max + padding, 300);
end

function limits = stable_plot_limits(varargin)
samples = [];
for input_index = 1:nargin
    samples = [samples; varargin{input_index}(:)];
end
samples = samples(isfinite(samples));
sample_min = min(samples);
sample_max = max(samples);
padding = max(0.08 * (sample_max - sample_min), 1e-3);
limits = [sample_min - padding, sample_max + padding];
end

function plot_density(samples, line_style, color, grid, bandwidth)
samples = samples(isfinite(samples));
if numel(samples) < 20 || std(samples) == 0, return; end
density = ksdensity(samples, grid, 'Bandwidth', bandwidth);
plot(grid, density, line_style, 'Color', color, 'LineWidth', 1.4);
end
