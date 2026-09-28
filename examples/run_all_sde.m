root = fileparts(fileparts(mfilename('fullpath')));
systems = {'dyad', 'tbh', 'l84', 'l96', 'cdv'};
for k = 1:numel(systems)
    run_isolated(fullfile(root, 'sde', [systems{k} '_stochastic_moi_vs_opinf_sde.m']));
end

function run_isolated(script_file)
rng('default');
run(script_file);
end
