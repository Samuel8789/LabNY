function [gFilter] = gaussian_filter(x0,y0,width, dimo)
w = width;
gFilter=zeros(dimo,dimo);
for col = 1 : w
  for row = 1 : w
    gFilter(row, col) = exp(-((col-x0)^2+(row-y0)^2)/w);
  end
end
end