function str = num2filename(val)
if floor(val)==val
    str = num2str(val);
    str(find(str=='-'))='n';
else
    str = num2str(val, '%1.1e');
    str(find(str=='.'))='p';
    str(find(str=='+'))='p';
    str(find(str=='-'))='n';
end
