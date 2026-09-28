function [theta_update, R_update, active] = em_stochastic_solver_sparse( ...
        u, res, r, J_num, theta_par, H_constraint, nonzero_index, tol, dt,...
        sparsity_thresh, Gamma, Niter_em, Niter_stlsq)

    if nargin < 10  || isempty(sparsity_thresh), sparsity_thresh = 0;   end
    if nargin < 11, Gamma = [];                                        end
    if nargin < 12 || isempty(Niter_em),        Niter_em    = 100;     end
    if nargin < 13 || isempty(Niter_stlsq),     Niter_stlsq = 10;      end

    p        = numel(nonzero_index);
    identity = speye(r);
    theta    = theta_par(:);

    Phi = cell(J_num, 1);
    for j = 1:J_num
        uj = u(:, j).';
        Mj = cat(2, kron(identity, kron(uj, uj)), kron(identity, uj));
        Phi{j} = Mj(:, nonzero_index);
    end

    if isempty(Gamma)
        Gmat = sparse(p, p);
    elseif isscalar(Gamma)
        Gmat = Gamma * speye(p);
    else
        Gmat = Gamma;
    end

    if isempty(H_constraint)
        protect = false(p, 1);
    else
        protect = full(any(H_constraint ~= 0, 1)).';
    end

    active = true(p, 1);

    for outer = 1:Niter_stlsq
        S = active;

        for it = 1:Niter_em
            R = zeros(r, r);
            for j = 1:J_num
                e = res(1:r, j) - Phi{j} * theta;
                R = R + e * e.';
            end
            R = diag(diag(R / J_num));

            nS = nnz(S);
            Ks = zeros(nS, nS);
            bs = zeros(nS, 1);
            for j = 1:J_num
                Pj  = Phi{j}(:, S);
                RiP = R \ Pj;
                Ks  = Ks + Pj.' * RiP;
                bs  = bs + Pj.' * (R \ res(1:r, j));
            end
            Ks = Ks + Gmat(S, S);

            if isempty(H_constraint)
                Hs = [];
            else
                Hs = H_constraint(:, S);
                Hs = Hs(any(Hs ~= 0, 2), :);
            end
            if isempty(Hs)
                theta_s = Ks \ bs;
            else
                KiHt    = Ks \ Hs.';
                KiB     = Ks \ bs;
                lam     = (Hs * KiHt) \ (Hs * KiB);
                theta_s = KiB - KiHt * lam;
            end

            theta_new      = zeros(p, 1);
            theta_new(S)   = theta_s;

            norm_iter = norm(theta_new - theta);
            theta     = theta_new;
            fprintf('  [pass %d] EM it %d, ||dtheta|| = %g\n', outer, it, norm_iter);
            if norm_iter <= tol
                break
            end
        end

        if sparsity_thresh <= 0
            break
        end
        new_active = (abs(theta) >= sparsity_thresh | protect) & active;
        if ~any(new_active)
            warning('em_stochastic_solver_sparse:emptySupport', ...
                    'All coefficients thresholded; reduce sparsity_thresh.');
            break
        end
        if isequal(new_active, active)
            break
        end
        active           = new_active;
        theta(~active)   = 0;
    end

    R = zeros(r, r);
    for j = 1:J_num
        e = res(1:r, j) - Phi{j} * theta;
        R = R + e * e.';
    end
    R_update     = diag(diag(R / J_num));
    theta_update = theta;
end
