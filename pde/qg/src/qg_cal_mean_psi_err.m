function err_mean_psi = qg_cal_mean_psi_err( ...
    par_qg, filenames, wr, r)

persistent cached_pod_path cached_mean_path POD_p psi_mean
if isempty(POD_p) || ~strcmp(cached_pod_path, filenames.pod_data)
    pod = load(filenames.pod_data, 'POD_p');
    POD_p = pod.POD_p;
    cached_pod_path = filenames.pod_data;
end
if isempty(psi_mean) || ~strcmp(cached_mean_path, filenames.mean_data)
    fom_mean = load(filenames.mean_data, 'psi_mean');
    psi_mean = fom_mean.psi_mean;
    cached_mean_path = filenames.mean_data;
end

psi_rom_mean = POD_p(:, 1:r) * mean(wr(1:r, :), 2);
diff_mean_psi = psi_mean - psi_rom_mean;
diff_mean_psi = reshape(diff_mean_psi, [par_qg.ny, par_qg.nx]);

err_mean_psi = sum(par_qg.W .* diff_mean_psi.^2, 'all');
end
