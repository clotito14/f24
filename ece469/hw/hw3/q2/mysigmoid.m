function G = mysigmoid(U,V)
gamma = 1;
c = -1;
G = tanh(gamma*U*V' + c);
end
