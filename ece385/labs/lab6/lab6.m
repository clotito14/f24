% make arrays for field current and stator voltage
ifield = [0.25, 0.5, 0.75, 1.0, 1.25];
vstator = [45, 81, 101, 113, 120];
% add airgap line
m = (81-45) / (0.5-0.25);
x = 0:0.01:1; 
y = m*x - m*0.25 + 45;

% open circuit characteristic
figure;
plot(x,y, 'r--');
hold on;
plot(ifield,vstator, 'b-');
hold off;

ylabel('Single-Phase Stator Voltage (V)');
xlabel('Field Current DC (A)');
title('Open Circuit Characteristic');
legend('Airgap Line', 'OCC');
grid on;

% short circuit characteristic

% make arrays for field current and stator voltage
ifield = [0, 0.3, 0.55, 1.05];
istator = [0, 0.425, 0.85, 1.7];

figure;
plot(ifield, istator, 'g-');
ylabel('Single-Phase Stator Current (A)');
xlabel('Field Current DC (A)');
title('Short Circuit Characteristic');
legend('SCC');
grid on;