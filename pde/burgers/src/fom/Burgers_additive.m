function Uvec = Burgers_additive(PAR, dW)

sigmas = PAR.sigmas;
lam = PAR.lam;
Grid_x = PAR.Grid_x;
dt = PAR.dt;
dx = PAR.dx;
SeignLap = PAR.SeignLap;
eigens = PAR.eigens;
Nmodes_dW = PAR.Nmodes_dW;
N = PAR.N;
t_skip = PAR.t_skip;

N_save = floor(N/t_skip)+1;
Uvec = zeros(Grid_x, N_save);
u_cur = PAR.U0;
Uvec(:,1) = u_cur;
fprintf('Solving the SPDE ... \n');

for t = 2:N
    if mod(t, 1e4) == 0
        fprintf(['    %d ... ','\n'], t);
    end
    u2 = u_cur.^2;
    u2_extend = [0;u2];
    u2_x = (u2_extend(2:end) - u2_extend(1:end-1))/dx;

    u2_x = - 0.5*PAR.gam*u2_x*dt;

    cur_dW = eigens(:,1:Nmodes_dW)*(dW(:,t-1).*sigmas);

    RHS = dst(u_cur + u2_x + cur_dW);

    LHS = (1-SeignLap - lam*dt);

    u_hat = RHS./LHS;
    u_next = idst(u_hat);
    u_cur = u_next;

    if mod(t, t_skip) == 0
    Uvec(:,floor(t/t_skip)+1) = u_cur;
    end
    if isnan(u_cur(2))
        warning('SPDE solution blew up')
        break;
    end
end

return;
