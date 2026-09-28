function ke = qg_cal_ke(wr, Er)

assert(size(Er, 1) == size(Er, 2), ...
    'The kinetic-energy operator must be square.');
assert(size(wr, 1) == size(Er, 1), ...
    'Reduced-state and kinetic-energy dimensions do not match.');

ke = 0.5 * sum(wr .* (Er * wr), 1);

roundoff_tol = 100 * eps(max(1, max(abs(ke))));
ke(ke < 0 & ke > -roundoff_tol) = 0;
assert(all(ke >= 0), 'Computed kinetic energy contains negative values.');
end
