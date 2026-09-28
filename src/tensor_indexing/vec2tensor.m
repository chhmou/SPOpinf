function [i,j,k] = vec2tensor(ind,r1,r2,r3)
if mod(ind, r1*r2)==0
    k = floor(ind/(r1*r2));
    j=r2;
    i=r1;
elseif mod(ind- floor(ind/(r1*r2))*(r1*r2),r1)==0
    k = floor(ind/(r1*r2))+1;
    j= floor((ind-(k-1)*r1*r2)/r1);
    i=r1;
else
        k = floor(ind/(r1*r2))+1;

        j= floor((ind-(k-1)*r1*r2)/r1)+1;

        i=ind-(k-1)*r1*r2-(j-1)*r1;

end

end
