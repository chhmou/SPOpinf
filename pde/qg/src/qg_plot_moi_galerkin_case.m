function qg_plot_moi_galerkin_case(result, case_dir)

r = result.r;
t = result.t_rom;
c_ref = [0.10 0.10 0.10];
c_gal = [0.00 0.35 0.70];
c_sde = [0.85 0.20 0.15];

ke_ref = qg_cal_ke(result.w_ref, result.Er);
fig_ke = figure('Color', 'w', 'Position', [100 80 1350 540]);
semilogy(t, max(ke_ref, eps), '-', 'Color', c_ref, ...
    'LineWidth', 1.5, 'DisplayName', 'FOM projection'); hold on
semilogy(t, max(result.metrics_gal.kinetic_energy, eps), '--', ...
    'Color', c_gal, 'LineWidth', 1.3, 'DisplayName', 'Galerkin ROM');
semilogy(t, max(result.metrics_moi_sde.kinetic_energy, eps), '-.', ...
    'Color', c_sde, 'LineWidth', 1.3, 'DisplayName', 'MOI SDE');
xlabel('Time'); ylabel('Kinetic energy'); grid on; box on
title(sprintf('QG kinetic energy comparison, r = %d', r));
legend('Location', 'best', 'NumColumns', 2);
exportgraphics(fig_ke, fullfile(case_dir, ...
    sprintf('kinetic_energy_r%02d.png', r)), 'Resolution', 300);
savefig(fig_ke, fullfile(case_dir, ...
    sprintf('kinetic_energy_r%02d.fig', r)));

modes = result.chosen_modes;
n_modes = numel(modes);
fig_modes = figure('Color', 'w', 'Position', [40 40 1900 1050]);
mode_layout = tiledlayout(n_modes, 3, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

time_window = t >= (t(1) + 0.5 * (t(end) - t(1)));
for row = 1:n_modes
    mode = modes(row);

    nexttile(3 * row - 2)
    plot(t(time_window), result.w_ref(mode, time_window), '-', ...
        'Color', c_ref, 'LineWidth', 1.1, 'DisplayName', 'FOM projection');
    hold on
    plot(t(time_window), result.wr_moi_sde(mode, time_window), '-.', ...
        'Color', c_sde, 'LineWidth', 1.0, 'DisplayName', 'MOI SDE');
    ylabel(sprintf('w_{%d}', mode)); grid on; box on
    if row == 1
        title('FOM and MOI SDE time series');
        legend('Location', 'best', 'NumColumns', 2);
    end
    if row == n_modes, xlabel('Time'); else, set(gca, 'XTickLabel', []); end

    nexttile(3 * row - 1)
    plot(t(time_window), result.wr_gal(mode, time_window), '--', ...
        'Color', c_gal, 'LineWidth', 1.0);
    ylabel(sprintf('w_{%d}', mode)); grid on; box on
    if row == 1, title('Galerkin-ROM time series'); end
    if row == n_modes, xlabel('Time'); else, set(gca, 'XTickLabel', []); end

    nexttile(3 * row)
    pdf_ref = result.w_ref(mode, :);
    pdf_gal = result.wr_gal(mode, :);
    pdf_sde = result.wr_moi_sde(mode, :);
    pdf_all = [pdf_ref, pdf_gal, pdf_sde];
    x_min = min(pdf_all);
    x_max = max(pdf_all);
    x_span = max(x_max - x_min, eps);
    x_pdf = linspace(x_min - 0.03 * x_span, ...
        x_max + 0.03 * x_span, 300);

    bandwidth = max(1.06 * std(pdf_ref) * numel(pdf_ref)^(-1/5), ...
        0.015 * x_span);
    bandwidth = max(bandwidth, sqrt(eps) * max(abs(pdf_all)));
    f_ref = ksdensity(pdf_ref, x_pdf, 'Bandwidth', bandwidth);
    f_gal = ksdensity(pdf_gal, x_pdf, 'Bandwidth', bandwidth);
    f_sde = ksdensity(pdf_sde, x_pdf, 'Bandwidth', bandwidth);
    plot(x_pdf, f_ref, '-', 'Color', c_ref, 'LineWidth', 1.3, ...
        'DisplayName', 'FOM projection'); hold on
    plot(x_pdf, f_gal, '--', 'Color', c_gal, 'LineWidth', 1.2, ...
        'DisplayName', 'Galerkin ROM');
    plot(x_pdf, f_sde, '-.', 'Color', c_sde, 'LineWidth', 1.2, ...
        'DisplayName', 'MOI SDE');
    ylabel('PDF'); grid on; box on
    if row == 1
        title('Kernel density estimate');
        legend('Location', 'best', 'NumColumns', 2);
    end
    if row == n_modes
        xlabel(sprintf('w_{%d}', mode));
    else
        set(gca, 'XTickLabel', []);
    end
end

title(mode_layout, sprintf( ...
    'QG selected-mode time series and PDFs, r = %d, modes = %s', ...
    r, mat2str(modes)));
exportgraphics(fig_modes, fullfile(case_dir, ...
    sprintf('selected_modes_timeseries_pdf_r%02d.png', r)), ...
    'Resolution', 300);
savefig(fig_modes, fullfile(case_dir, ...
    sprintf('selected_modes_timeseries_pdf_r%02d.fig', r)));
end
