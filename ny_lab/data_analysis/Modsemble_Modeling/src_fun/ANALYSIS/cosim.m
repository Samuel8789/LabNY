function [CS] = cosim(A,B)
CS = sum(A.*B)/(sqrt((sum(A.^2)*sum(B.^2))));
end