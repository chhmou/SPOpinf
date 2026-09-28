function [B, C, D] = qg_galerkin_to_polynomial( ...
    par_qg, qg_operator, r)

qg_nonlin = qg_operator.Txy - qg_operator.Tyx;
qg_lin = (1 / par_qg.Ro) * qg_operator.Mr_x - ...
    (1 / par_qg.Re) * qg_operator.Sr;
qg_force = (1 / par_qg.Ro) * qg_operator.fr;

C = qg_operator.Mr \ qg_lin;
D = qg_operator.Mr \ qg_force;

quadratic_full = zeros(r, r, r);
mass_inverse = qg_operator.Mr \ eye(r);
for output = 1:r
    for projected_output = 1:r
        quadratic_full(:, :, output) = ...
            quadratic_full(:, :, output) + ...
            mass_inverse(output, projected_output) * ...
            qg_nonlin(:, :, projected_output);
    end
end

B = zeros(r, r, r);
for output = 1:r
    for i = 1:r
        B(i, i, output) = quadratic_full(i, i, output);
        for j = i + 1:r
            B(i, j, output) = quadratic_full(i, j, output) + ...
                quadratic_full(j, i, output);
        end
    end
end
end
