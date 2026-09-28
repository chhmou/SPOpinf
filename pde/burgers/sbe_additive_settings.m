function [sbe_settings, filenames] = sbe_additive_settings(project_root, data_root)

if nargin < 1 || isempty(project_root)
    project_root = fileparts(mfilename('fullpath'));
end
if nargin < 2 || isempty(data_root) || strcmp(data_root, '<path to data>')
    error('sbe_additive_settings:dataPath', ...
        'Set dataPath at the top of run_sbe_pathwise_comparison.m to the folder that contains test3p0en03/.');
end

L = 2*pi;
nu = 0.005;
lam = 9*nu*pi^2/L^2;
sigma_par = 0.003;

test_name = ['test', num2filename(sigma_par)];
input_folder = fullfile(data_root, test_name);
assert(exist(input_folder, 'dir') == 7, ...
    'Legacy SBE input directory not found: %s', input_folder);

fom_name = ['sbe_fom_data_sig', num2filename(sigma_par), ...
    '_nu', num2filename(nu), '_lam', num2filename(lam), '.mat'];
dW_name = ['sbe_noise_dw_sig', num2filename(sigma_par), '.mat'];
ke_name = ['sbe_ke_data_sig', num2filename(sigma_par), ...
    '_nu', num2filename(nu), '_lam', num2filename(lam), '.mat'];
proj_name = ['sbe_proj_data_sig', num2filename(sigma_par), ...
    '_nu', num2filename(nu), '_lam', num2filename(lam), '.mat'];

fom_data = fullfile(input_folder, fom_name);
dW_data = fullfile(input_folder, dW_name);
ke_data = fullfile(input_folder, ke_name);
fom_proj_data = fullfile(input_folder, proj_name);

required_files = {fom_data, dW_data, fom_proj_data};
for file_index = 1:numel(required_files)
    assert(exist(required_files{file_index}, 'file') == 2, ...
        'Required legacy input not found: %s', required_files{file_index});
end

loaded_fom = load(fom_data, 'PAR');
PAR = loaded_fom.PAR;

results_root = fullfile(project_root, 'results');
case_root = fullfile(results_root, test_name);
plot_path = fullfile(case_root, 'figures');
if ~exist(case_root, 'dir'), mkdir(case_root); end
if ~exist(plot_path, 'dir'), mkdir(plot_path); end

sbe_settings = struct('PAR', PAR);
filenames = struct( ...
    'data_root', data_root, ...
    'input_folder', input_folder, ...
    'results_root', results_root, ...
    'folder_name', case_root, ...
    'fom_data', fom_data, ...
    'dW_data', dW_data, ...
    'ke_data', ke_data, ...
    'fom_proj_data', fom_proj_data, ...
    'plot_path', plot_path);
end
