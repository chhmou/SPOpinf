function [A,B] = burgers_sine_nonlinear(PAR,modes)

L = PAR.L;

A = diag((pi^2/L^2).*(1:modes).^2);

B = zeros(modes,modes,modes);

coeff = (pi/4)*(2/L)^(3/2);

for i = 1:modes
    for j =1:modes
        for k = 1:modes

                        if i+j==k

                            B(i,j,k) = coeff*j;
                        elseif i-j==k
                            B(i,j,k) = coeff*j;

                        elseif i-j==-k
                            B(i,j,k) = -coeff*j;
                        else
                            B(i,j,k) = 0;
                        end

        end
    end
end

end
