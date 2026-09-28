root = fileparts(fileparts(mfilename('fullpath')));
qg_dir = fullfile(root, 'pde', 'qg');
run_isolated(fullfile(qg_dir, 'qg_moi_vs_galerkin_r_sweep.m'));
run_isolated(fullfile(qg_dir, 'qg_stochastic_closure_vs_moi_r_sweep.m'));

function run_isolated(script_file)
run(script_file);
end
