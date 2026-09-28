function dw = qg_polynomial_drift(w, B, C, D)

r = numel(w);
ww = w * w';
dw = C * w + D;
for output = 1:r
    dw(output) = dw(output) + sum(B(:, :, output) .* ww, 'all');
end
end
