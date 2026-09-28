function [Cw_drift, Cw_sde] = qg_moi_rom_fun( ...
    par_qg, B, C, D, sigma, wr0, rng_seed)

r = numel(wr0);
N_save = par_qg.n_time / par_qg.save_index;
Cw_drift = zeros(r, N_save + 1);
Cw_sde = zeros(r, N_save + 1);
Cw_drift(:, 1) = wr0;
Cw_sde(:, 1) = wr0;

w_drift = wr0;
w_sde = wr0;
dt = par_qg.ddt;
rng(rng_seed);
save_count = 1;

fprintf('Integrating MOI drift and stochastic MOI SDE ...\n');
for step = 1:par_qg.n_time
    k1 = qg_polynomial_drift(w_drift, B, C, D);
    k2 = qg_polynomial_drift(w_drift + 0.5 * dt * k1, B, C, D);
    k3 = qg_polynomial_drift(w_drift + 0.5 * dt * k2, B, C, D);
    k4 = qg_polynomial_drift(w_drift + dt * k3, B, C, D);
    w_drift = w_drift + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);

    w_sde = w_sde + qg_polynomial_drift(w_sde, B, C, D) * dt + ...
        sigma .* (sqrt(dt) * randn(r, 1));

    if any(~isfinite(w_drift)) || any(~isfinite(w_sde)) || ...
            norm(w_drift) > 1e16 || norm(w_sde) > 1e16
        error('qg_moi_rom_fun:IntegrationDiverged', ...
            'The learned MOI model diverged at t = %.6g.', ...
            par_qg.t0 + step * dt);
    end

    if rem(step, par_qg.save_index) == 0
        save_count = save_count + 1;
        Cw_drift(:, save_count) = w_drift;
        Cw_sde(:, save_count) = w_sde;
    end
end
fprintf('MOI integrations completed.\n');
end
