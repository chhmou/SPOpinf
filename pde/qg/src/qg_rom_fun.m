function [Cw] = qg_rom_fun(par_qg, filenames, qg_operator, wr0, r)

if nargin < 2 || ~isstruct(filenames)
    error('qg_rom_fun:InvalidFilenames', ...
        'filenames must be the configuration file structure.');
end

validateattributes(wr0, {'double'}, {'column', 'numel', r, 'finite'});
assert(par_qg.n_time / par_qg.save_index == ...
    round(par_qg.n_time / par_qg.save_index), ...
    'n_time must be divisible by save_index.');

Ro = par_qg.Ro;
Re = par_qg.Re;
ddt = par_qg.ddt;
save_index = par_qg.save_index;

Mr = qg_operator.Mr;
qg_lin = (1 / Ro) * qg_operator.Mr_x - ...
    (1 / Re) * qg_operator.Sr;
qg_force = (1 / Ro) * qg_operator.fr;
qg_nonlin = qg_operator.Txy - qg_operator.Tyx;

N_save = par_qg.n_time / save_index;
Cw = zeros(r, N_save + 1);
Cw(:, 1) = wr0;

fprintf('Integrating the r = %d Galerkin ROM with RK4 ...\n', r);
save_count = 1;

for k = 1:par_qg.n_time
    K1 = rom_rhs(wr0, Mr, qg_lin, qg_force, qg_nonlin, r);
    K2 = rom_rhs(wr0 + 0.5 * ddt * K1, ...
        Mr, qg_lin, qg_force, qg_nonlin, r);
    K3 = rom_rhs(wr0 + 0.5 * ddt * K2, ...
        Mr, qg_lin, qg_force, qg_nonlin, r);
    K4 = rom_rhs(wr0 + ddt * K3, ...
        Mr, qg_lin, qg_force, qg_nonlin, r);

    w_new = wr0 + (ddt / 6) * (K1 + 2 * K2 + 2 * K3 + K4);
    if any(~isfinite(w_new)) || norm(w_new) > 1e16
        error('qg_rom_fun:IntegrationDiverged', ...
            'The Galerkin ROM diverged at t = %.6g.', ...
            par_qg.t0 + k * ddt);
    end

    if rem(k, save_index) == 0
        save_count = save_count + 1;
        Cw(:, save_count) = w_new;
    end
    wr0 = w_new;
end

fprintf('Galerkin-ROM integration completed.\n');
end

function dw = rom_rhs(w, Mr, qg_lin, qg_force, qg_nonlin, r)
nonlinear_term = qg_jacobian_rom(w, qg_nonlin, r);
dw = Mr \ (nonlinear_term + qg_lin * w + qg_force);
end
