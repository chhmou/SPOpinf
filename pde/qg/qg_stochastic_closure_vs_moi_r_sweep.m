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
options = struct( ...
    'n_training_pairs', 400, ...
    'rng_seed', 2026, ...
    'tol', 1e-6, ...
    'sparsity_thresh', 1e-3, ...
    'Gamma', 1e-8, ...
    'Niter_em', 20, ...
    'Niter_stlsq', 5, ...
    'use_global_state_scale', false);

direct_root = fullfile(project_dir, ...
    'results_qg_moi_vs_galerkin_r_sweep');
output_root = fullfile(project_dir, ...
    'results_qg_stochastic_closure_vs_moi_r_sweep');
if ~exist(output_root, 'dir'), mkdir(output_root); end
diary(fullfile(output_root, 'sweep_log.txt')); diary on

fprintf('\n================================================================\n');
fprintf('  QG CPOpInf-EC vs G-ROM and CPOpInf-NC\n');
fprintf('  closure target = da/dt - f_Galerkin(a)\n');
fprintf('  r values: %s\n', mat2str(r_values));
fprintf('  common adjacent training pairs per r: %d\n', ...
    options.n_training_pairs);
fprintf('================================================================\n\n');

r_max = max(r_values);
operator_file = fullfile(filenames.cache_root, ...
    sprintf('qg_galerkin_operators_r%d.mat', r_max));
cached_operator = load(operator_file, 'qg_operator');
qg_operator_max = cached_operator.qg_operator;
assert(isfield(qg_operator_max, 'Er'), ...
    'The cached r=%d Galerkin operator is missing Er.', r_max);

n_cases = numel(r_values);
status = strings(n_cases, 1);
runtime_seconds = NaN(n_cases, 1);
mean_l2_gal = NaN(n_cases, 1);
mean_l2_moi = NaN(n_cases, 1);
mean_l2_closure = NaN(n_cases, 1);

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

    case_dir = fullfile(output_root, sprintf('r_%02d', r));
    if ~exist(case_dir, 'dir'), mkdir(case_dir); end
    case_file = fullfile(case_dir, 'results.mat');
    direct_file = fullfile(direct_root, sprintf('r_%02d', r), 'results.mat');

    fprintf('\n################################################################\n');
    fprintf('  STOCHASTIC CLOSURE r = %d (%d of %d)\n', ...
        r, case_index, n_cases);
    fprintf('################################################################\n');
    timer = tic;

    try
        assert(exist(direct_file, 'file') == 2, ...
            'Required direct-MOI checkpoint not found: %s', direct_file);
        direct_checkpoint = load(direct_file, 'case_result');
        direct_result = direct_checkpoint.case_result;

        if exist(case_file, 'file')
            checkpoint = load(case_file, 'case_result');
            if isfield(checkpoint.case_result, 'method_version') && ...
                    checkpoint.case_result.method_version == 2
                case_result = checkpoint.case_result;
                status(case_index) = "reused";
                fprintf('Loaded completed checkpoint: %s\n', case_file);
            else
                fprintf('Refreshing outdated checkpoint: %s\n', case_file);
                qg_operator = qg_slice_operator(qg_operator_max, r);
                case_result = qg_run_stochastic_closure_case( ...
                    cfg, w_proj, qg_operator, direct_result, r, case_options);
                save(case_file, 'case_result', 'case_options');
                status(case_index) = "refreshed";
            end
        else
            qg_operator = qg_slice_operator(qg_operator_max, r);
            case_result = qg_run_stochastic_closure_case( ...
                cfg, w_proj, qg_operator, direct_result, r, case_options);
            save(case_file, 'case_result', 'case_options');
            status(case_index) = "completed";
        end

        qg_print_stochastic_closure_summary(case_result);
        qg_plot_stochastic_closure_case(case_result, case_dir);

        mean_l2_gal(case_index) = ...
            case_result.metrics_gal.mean_streamfunction_l2;
        mean_l2_moi(case_index) = ...
            case_result.metrics_moi_sde.mean_streamfunction_l2;
        mean_l2_closure(case_index) = ...
            case_result.metrics_closure_sde.mean_streamfunction_l2;
        runtime_seconds(case_index) = toc(timer);
        fprintf('Finished closure r = %d in %.2f seconds.\n', ...
            r, runtime_seconds(case_index));
    catch exception
        runtime_seconds(case_index) = toc(timer);
        status(case_index) = "failed: " + string(exception.identifier);
        fprintf(2, 'FAILED closure r = %d after %.2f seconds:\n%s\n', ...
            r, runtime_seconds(case_index), ...
            getReport(exception, 'extended', 'hyperlinks', 'off'));
    end
    close all

    comparison_table = table(r_values(:), mean_l2_gal, mean_l2_moi, ...
        mean_l2_closure, runtime_seconds, status, ...
        'VariableNames', {'r', 'GROM', 'CPOpInf_NC', 'CPOpInf_EC', ...
        'RuntimeSeconds', 'Status'});
    save(fullfile(output_root, 'sweep_summary.mat'), ...
        'comparison_table', 'options');
    writetable(comparison_table, fullfile(output_root, ...
        'mean_streamfunction_l2.csv'));
end

fprintf('\n================================================================\n');
fprintf('  MEAN-STREAMFUNCTION L2: G-ROM vs CPOpInf-NC vs CPOpInf-EC\n');
fprintf('================================================================\n');
fprintf('%6s  %14s  %14s  %14s\n', ...
    'r', 'G-ROM', 'CPOpInf-NC', 'CPOpInf-EC');
for case_index = 1:n_cases
    fprintf('%6d  %14.6e  %14.6e  %14.6e\n', ...
        r_values(case_index), mean_l2_gal(case_index), ...
        mean_l2_moi(case_index), mean_l2_closure(case_index));
end
fprintf('================================================================\n');
fprintf('Saved stochastic-closure outputs in: %s%s\n\n', ...
    output_root, filesep);
disp(comparison_table)
diary off
