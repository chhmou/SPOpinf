root = fileparts(fileparts(mfilename('fullpath')));
systems = {'dyad', 'tbh', 'l84', 'l96', 'cdv'};
for k = 1:numel(systems)
    run_isolated(fullfile(root, 'sde', 'data_generation', ['generate_' systems{k} '_data.m']));
end

function run_isolated(script_file)
rng('default');
run(script_file);
end
