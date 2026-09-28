function Er = qg_rom_energy_operator(par_qg, filenames, r)

pod = load(filenames.pod_data, 'POD_p');
assert(size(pod.POD_p, 2) >= r, ...
    'The POD file does not contain the requested %d modes.', r);

N_dof = par_qg.N_dof;
pPODx = zeros(N_dof, r);
pPODy = zeros(N_dof, r);

fprintf('Assembling the reduced kinetic-energy operator ...\n');
for i = 1:r
    p_mode = reshape(pod.POD_p(:, i), [par_qg.ny, par_qg.nx]);
    [p_x, p_y] = compute_velocity(p_mode, par_qg.dx, par_qg.dy, ...
        par_qg.nx, par_qg.ny);
    pPODx(:, i) = p_x(:);
    pPODy(:, i) = p_y(:);
end

mass_weight = spdiags(par_qg.weight, 0, N_dof, N_dof);
Er = pPODx' * mass_weight * pPODx + ...
    pPODy' * mass_weight * pPODy;
Er = 0.5 * (Er + Er');
end
