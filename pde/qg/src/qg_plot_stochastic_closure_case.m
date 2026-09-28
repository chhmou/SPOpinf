function qg_plot_stochastic_closure_case(result, case_dir)

r = result.r;
t = result.t_rom;
c_ref = [0.10 0.10 0.10];
c_gal = [0.00 0.35 0.70];
c_moi = [0.85 0.20 0.15];
c_closure = [0.45 0.15 0.70];

ke_ref = qg_cal_ke(result.w_ref, result.Er);
fig_ke = figure('Color', 'w', 'Position', [100 80 1400 560]);
semilogy(t, max(ke_ref, eps), '-', 'Color', c_ref, ...
    'LineWidth', 1.5, 'DisplayName', 'FOM projection'); hold on
semilogy(t, max(result.metrics_gal.kinetic_energy, eps), '--', ...
    'Color', c_gal, 'LineWidth', 1.3, 'DisplayName', 'Galerkin ROM');
semilogy(t, max(result.metrics_moi_sde.kinetic_energy, eps), '-.', ...
    'Color', c_moi, 'LineWidth', 1.2, 'DisplayName', 'CPOpInf-NC');
semilogy(t, max(result.metrics_closure_sde.kinetic_energy, eps), ':', ...
    'Color', c_closure, 'LineWidth', 1.5, ...
    'DisplayName', 'CPOpInf-EC');
xlabel('Time'); ylabel('Kinetic energy'); grid on; box on
title(sprintf('QG kinetic energy: G-ROM and CPOpInf models, r = %d', r));
legend('Location', 'best', 'NumColumns', 2);
exportgraphics(fig_ke, fullfile(case_dir, ...
    sprintf('kinetic_energy_closure_comparison_r%02d.png', r)), ...
    'Resolution', 300);
savefig(fig_ke, fullfile(case_dir, ...
    sprintf('kinetic_energy_closure_comparison_r%02d.fig', r)));

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
        'Color', c_moi, 'LineWidth', 1.0, 'DisplayName', 'CPOpInf-NC');
    plot(t(time_window), result.wr_closure_sde(mode, time_window), ':', ...
        'Color', c_closure, 'LineWidth', 1.2, ...
        'DisplayName', 'CPOpInf-EC');
    ylabel(sprintf('w_{%d}', mode)); grid on; box on
    if row == 1
        title('FOM and CPOpInf time series');
        legend('Location', 'best', 'NumColumns', 3);
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
    pdf_moi = result.wr_moi_sde(mode, :);
    pdf_closure = result.wr_closure_sde(mode, :);
    pdf_all = [pdf_ref, pdf_gal, pdf_moi, pdf_closure];
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
    f_moi = ksdensity(pdf_moi, x_pdf, 'Bandwidth', bandwidth);
    f_closure = ksdensity(pdf_closure, x_pdf, 'Bandwidth', bandwidth);
    plot(x_pdf, f_ref, '-', 'Color', c_ref, 'LineWidth', 1.3, ...
        'DisplayName', 'FOM projection'); hold on
    plot(x_pdf, f_gal, '--', 'Color', c_gal, 'LineWidth', 1.2, ...
        'DisplayName', 'Galerkin ROM');
    plot(x_pdf, f_moi, '-.', 'Color', c_moi, 'LineWidth', 1.2, ...
        'DisplayName', 'CPOpInf-NC');
    plot(x_pdf, f_closure, ':', 'Color', c_closure, 'LineWidth', 1.5, ...
        'DisplayName', 'CPOpInf-EC');
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
    'QG CPOpInf-NC and CPOpInf-EC comparison, r = %d, modes = %s', ...
    r, mat2str(modes)));
exportgraphics(fig_modes, fullfile(case_dir, ...
    sprintf('selected_modes_closure_comparison_r%02d.png', r)), ...
    'Resolution', 300);
savefig(fig_modes, fullfile(case_dir, ...
    sprintf('selected_modes_closure_comparison_r%02d.fig', r)));
end
