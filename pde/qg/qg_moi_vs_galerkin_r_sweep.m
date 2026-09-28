clear; clc; close all

dataPath = '<path to data>';

project_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(project_dir, 'src'));
addpath(genpath(fullfile(project_dir, '..', '..', 'src')));

cfg = qg_galerkin_config(project_dir, dataPath);
par_qg = cfg.par_qg;
filenames = cfg.filenames;
projection = load(filenames.fom_proj_data, 'w_proj');
w_proj = projection.w_proj;

r_values = [5 10 15 20 25 30];

options = struct;
options.n_training_pairs = 400;
options.rng_seed = 2026;
options.tol = 1e-6;
options.sparsity_thresh = 1e-3;
options.Gamma = 1e-8;
options.Niter_em = 20;
options.Niter_stlsq = 5;
options.use_global_state_scale = false;

sweep_root = fullfile(project_dir, 'results_qg_moi_vs_galerkin_r_sweep');
if ~exist(sweep_root, 'dir'), mkdir(sweep_root); end
diary_file = fullfile(sweep_root, 'sweep_log.txt');
diary(diary_file); diary on

fprintf('\n================================================================\n');
fprintf('  QG MOI-vs-Galerkin dimension sweep\n');
fprintf('  r values: %s\n', mat2str(r_values));
fprintf('  common adjacent training pairs per r: %d\n', ...
    options.n_training_pairs);
fprintf('  optimizer: unchanged em_stochastic_solver_sparse\n');
fprintf('================================================================\n\n');

r_max = max(r_values);
operator_file = fullfile(filenames.cache_root, ...
    sprintf('qg_galerkin_operators_r%d.mat', r_max));
if exist(operator_file, 'file')
    cached = load(operator_file, 'qg_operator');
    qg_operator_max = cached.qg_operator;
else
    qg_operator_max = qg_rom_operator(cfg, filenames, r_max);
    qg_operator = qg_operator_max;
    operator_metadata = struct('r', r_max, 'Re', par_qg.Re, ...
        'Ro', par_qg.Ro, 'N_snap', par_qg.N_snap, ...
        'pod_data', filenames.pod_data);
    save(operator_file, 'qg_operator', 'operator_metadata');
end
if ~isfield(qg_operator_max, 'Er')
    qg_operator_max.Er = qg_rom_energy_operator( ...
        par_qg, filenames, r_max);
    qg_operator = qg_operator_max;
    if exist('operator_metadata', 'var')
        save(operator_file, 'qg_operator', 'operator_metadata');
    else
        save(operator_file, 'qg_operator', '-append');
    end
end

n_cases = numel(r_values);
case_status = strings(n_cases, 1);
case_runtime_seconds = NaN(n_cases, 1);
mean_psi_l2_gal = NaN(n_cases, 1);
mean_psi_l2_moi_drift = NaN(n_cases, 1);
mean_psi_l2_moi_sde = NaN(n_cases, 1);

for case_index = 1:n_cases
    r = r_values(case_index);
    case_options = options;
    if r == 25
        case_options.Niter_em = 10;
        case_options.Niter_stlsq = 1;
    elseif r == 30
        case_options.Niter_em = 1;
        case_options.Niter_stlsq = 1;
        case_options.use_global_state_scale = true;
        case_options.Gamma = 1e-1;
    end
    case_dir = fullfile(sweep_root, sprintf('r_%02d', r));
    if ~exist(case_dir, 'dir'), mkdir(case_dir); end
    case_file = fullfile(case_dir, 'results.mat');

    fprintf('\n################################################################\n');
    fprintf('  DIMENSION r = %d (%d of %d)\n', r, case_index, n_cases);
    fprintf('################################################################\n');

    timer = tic;
    try
        if exist(case_file, 'file')
            saved_case = load(case_file, 'case_result');
            case_result = saved_case.case_result;
            fprintf('Loaded completed checkpoint: %s\n', case_file);
            case_status(case_index) = "reused";
        else
            qg_operator = qg_slice_operator(qg_operator_max, r);
            case_result = qg_run_moi_galerkin_case( ...
                cfg, w_proj, qg_operator, r, case_options);
            save(case_file, 'case_result', 'case_options');
            case_status(case_index) = "completed";
        end

        qg_print_moi_galerkin_summary(case_result);
        qg_plot_moi_galerkin_case(case_result, case_dir);

        mean_psi_l2_gal(case_index) = ...
            case_result.metrics_gal.mean_streamfunction_l2;
        mean_psi_l2_moi_drift(case_index) = ...
            case_result.metrics_moi_drift.mean_streamfunction_l2;
        mean_psi_l2_moi_sde(case_index) = ...
            case_result.metrics_moi_sde.mean_streamfunction_l2;
        case_runtime_seconds(case_index) = toc(timer);

        fprintf('Finished r = %d in %.2f seconds.\n', ...
            r, case_runtime_seconds(case_index));
    catch exception
        case_runtime_seconds(case_index) = toc(timer);
        case_status(case_index) = "failed: " + string(exception.identifier);
        fprintf(2, 'FAILED r = %d after %.2f seconds:\n%s\n', ...
            r, case_runtime_seconds(case_index), ...
            getReport(exception, 'extended', 'hyperlinks', 'off'));
    end

    close all

    mean_streamfunction_l2_table = table(r_values(:), ...
        mean_psi_l2_gal, mean_psi_l2_moi_drift, mean_psi_l2_moi_sde, ...
        case_runtime_seconds, case_status, ...
        'VariableNames', {'r', 'GROM', 'MOI_Drift', 'MOI_SDE', ...
        'RuntimeSeconds', 'Status'});
    save(fullfile(sweep_root, 'sweep_summary.mat'), ...
        'mean_streamfunction_l2_table', 'options');
    writetable(mean_streamfunction_l2_table, ...
        fullfile(sweep_root, 'mean_streamfunction_l2.csv'));
end

fprintf('\n================================================================\n');
fprintf('  MEAN-STREAMFUNCTION L2 ERROR FOR DIFFERENT r\n');
fprintf('================================================================\n');
fprintf('%6s  %14s  %14s  %14s\n', ...
    'r', 'G-ROM', 'MOI drift', 'MOI SDE');
fprintf('%6s  %14s  %14s  %14s\n', ...
    '------', '--------------', '--------------', '--------------');
for case_index = 1:n_cases
    fprintf('%6d  %14.6e  %14.6e  %14.6e\n', ...
        r_values(case_index), mean_psi_l2_gal(case_index), ...
        mean_psi_l2_moi_drift(case_index), ...
        mean_psi_l2_moi_sde(case_index));
end
fprintf('================================================================\n');
fprintf('Saved sweep outputs in: %s%s\n\n', sweep_root, filesep);
disp(mean_streamfunction_l2_table)
diary off
