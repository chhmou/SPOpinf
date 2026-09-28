root = fileparts(fileparts(mfilename('fullpath')));
for r = [2 4 6]
    setenv('SBE_R', num2str(r));
    run_isolated(fullfile(root, 'pde', 'burgers', 'run_sbe_pathwise_comparison.m'));
end
setenv('SBE_R', '');

function run_isolated(script_file)
run(script_file);
end
