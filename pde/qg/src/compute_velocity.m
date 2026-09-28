function [px, py] = compute_velocity(p, dx, dy, xdim, ydim)

assert(isequal(size(p), [ydim, xdim]), ...
    'Input field size must be [ydim, xdim].');

px = zeros(ydim, xdim);
py = zeros(ydim, xdim);

for i = 1:ydim
    for j = 1:xdim
        if i <= 2
            py(i, j) = (-25 * p(i, j) / 12 + 4 * p(i + 1, j) ...
                - 3 * p(i + 2, j) + 4 * p(i + 3, j) / 3 ...
                - p(i + 4, j) / 4) / dy;
        elseif i >= ydim - 1
            py(i, j) = (25 * p(i, j) / 12 - 4 * p(i - 1, j) ...
                + 3 * p(i - 2, j) - 4 * p(i - 3, j) / 3 ...
                + p(i - 4, j) / 4) / dy;
        else
            py(i, j) = (p(i - 2, j) / 12 - 2 * p(i - 1, j) / 3 ...
                + 2 * p(i + 1, j) / 3 - p(i + 2, j) / 12) / dy;
        end

        if j <= 2
            px(i, j) = (-25 * p(i, j) / 12 + 4 * p(i, j + 1) ...
                - 3 * p(i, j + 2) + 4 * p(i, j + 3) / 3 ...
                - p(i, j + 4) / 4) / dx;
        elseif j >= xdim - 1
            px(i, j) = (25 * p(i, j) / 12 - 4 * p(i, j - 1) ...
                + 3 * p(i, j - 2) - 4 * p(i, j - 3) / 3 ...
                + p(i, j - 4) / 4) / dx;
        else
            px(i, j) = (p(i, j - 2) / 12 - 2 * p(i, j - 1) / 3 ...
                + 2 * p(i, j + 1) / 3 - p(i, j + 2) / 12) / dx;
        end
    end
end
end
