function G = mysigmoid_2(U,V)
gamma = 0.5;
c = -1;
G = tanh(gamma*U*V' + c);
end
