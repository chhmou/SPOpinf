function wr_closure_sde = qg_stochastic_closure_rom_fun( ...
    par_qg, B_gal, C_gal, D_gal, closure, wr0, rng_seed)

r = numel(wr0);
n_save = par_qg.n_time / par_qg.save_index;
wr_closure_sde = zeros(r, n_save + 1);
wr_closure_sde(:, 1) = wr0;

w = wr0;
dt = par_qg.ddt;
rng(rng_seed);
save_count = 1;

fprintf('Integrating stochastic closure ROM ...\n');
for step = 1:par_qg.n_time
    drift = qg_polynomial_drift(w, B_gal, C_gal, D_gal) + ...
        qg_polynomial_drift(w, closure.B, closure.C, closure.D);
    w = w + drift * dt + closure.sigma .* (sqrt(dt) * randn(r, 1));

    if any(~isfinite(w)) || norm(w) > 1e16
        error('qg_stochastic_closure_rom_fun:IntegrationDiverged', ...
            'The stochastic closure ROM diverged at t = %.6g.', ...
            par_qg.t0 + step * dt);
    end

    if rem(step, par_qg.save_index) == 0
        save_count = save_count + 1;
        wr_closure_sde(:, save_count) = w;
    end
end
fprintf('Stochastic closure-ROM integration completed.\n');
end
