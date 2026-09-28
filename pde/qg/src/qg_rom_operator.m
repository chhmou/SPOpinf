function qg_operator = qg_rom_operator(qg_settings, filenames, r)

par_qg = qg_settings.par_qg;
pod = load(filenames.pod_data, 'POD_p', 'POD_w');

assert(size(pod.POD_w, 2) >= r && size(pod.POD_p, 2) >= r, ...
    'The POD file does not contain the requested %d modes.', r);

N_dof = par_qg.N_dof;
nx = par_qg.nx;
ny = par_qg.ny;
dx = par_qg.dx;
dy = par_qg.dy;
weight = par_qg.weight;

[~, Y] = meshgrid(par_qg.xgrid, par_qg.ygrid);
force = sin(pi * (Y - 1));
force = force(:);

pPODx = zeros(N_dof, r);
pPODy = zeros(N_dof, r);
wPODx = zeros(N_dof, r);
wPODy = zeros(N_dof, r);

fprintf('Computing spatial derivatives of %d POD modes ...\n', r);
for i = 1:r
    p_mode = reshape(pod.POD_p(:, i), [ny, nx]);
    [p_x, p_y] = compute_velocity(p_mode, dx, dy, nx, ny);
    pPODx(:, i) = p_x(:);
    pPODy(:, i) = p_y(:);

    w_mode = reshape(pod.POD_w(:, i), [ny, nx]);
    [w_x, w_y] = compute_velocity(w_mode, dx, dy, nx, ny);
    wPODx(:, i) = w_x(:);
    wPODy(:, i) = w_y(:);
end

mass_weight = spdiags(weight, 0, N_dof, N_dof);
POD_w_r = pod.POD_w(:, 1:r);

Mr = POD_w_r' * mass_weight * POD_w_r;
Sr = wPODx' * mass_weight * wPODx + ...
    wPODy' * mass_weight * wPODy;
Er = pPODx' * mass_weight * pPODx + ...
    pPODy' * mass_weight * pPODy;
Mr_x = POD_w_r' * mass_weight * pPODx;
fr = POD_w_r' * mass_weight * force;

Tyx = zeros(r, r, r);
Txy = zeros(r, r, r);
fprintf('Assembling the quadratic Galerkin tensor ...\n');
for k = 1:r
    weighted_mode = weight .* POD_w_r(:, k);
    Tyx(:, :, k) = pPODy' * (weighted_mode .* wPODx);
    Txy(:, :, k) = pPODx' * (weighted_mode .* wPODy);
end

qg_operator = struct( ...
    'Mr', Mr, ...
    'Mr_x', Mr_x, ...
    'Sr', Sr, ...
    'Er', Er, ...
    'fr', fr, ...
    'Tyx', Tyx, ...
    'Txy', Txy);
end
