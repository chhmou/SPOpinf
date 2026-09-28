function Nh = qg_jacobian_rom(C0, Ten, r)

Nh = zeros(r, 1);
for i = 1:r
    Nh(i) = C0' * Ten(:, :, i) * C0;
end
end
