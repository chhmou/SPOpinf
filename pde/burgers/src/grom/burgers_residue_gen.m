function [residue] = burgers_residue_gen(PAR,dW, u_proj,A,B,r)

N_train  = size(u_proj,2)-1;

residue = zeros(r,N_train);
sigmas = PAR.sigmas;
lam = PAR.lam;
dt = PAR.dt*PAR.t_skip;
nu = PAR.nu;

B_var = B;

for ii = 1:N_train

    u0 = u_proj(1:r,ii);
    u1 = u_proj(1:r,ii+1);

    dudt = (u1-u0)./dt;

    rhs_lin = (lam.*eye(r)-nu.*A)*u1;

    rhs_nonlin = zeros(r,1);

    for j=1:r
        rhs_nonlin(j) = u0'*B_var(:,:,j)*u0;
    end

    residue(:,ii) = dudt- rhs_lin + PAR.gam.*rhs_nonlin;

end

return
