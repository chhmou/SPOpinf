function rel_entropy = qg_cal_relative_entropy(a_ref, a_simul)

[pdf_ref, x_ref] = ksdensity(a_ref);
[pdf_simul, x_simul] = ksdensity(a_simul);

x_common = sort([x_ref, x_simul], 'ascend');
pdf_ref_interp = interp1(x_ref, pdf_ref, x_common, 'spline');
pdf_simul_interp = interp1(x_simul, pdf_simul, x_common, 'spline');

positive = pdf_ref_interp .* pdf_simul_interp > 0;
integrand = pdf_ref_interp(positive) .* log( ...
    pdf_ref_interp(positive) ./ pdf_simul_interp(positive));
rel_entropy = sum(integrand);
end
