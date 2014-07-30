% simple implementation of the Usher-McClelland model for perceptual
% classification 
% model from Usher, McClelland, On the Time Course of Perceptual Choice:The
% Leaky Competing Accumulator Model, 2001

% figure
close all; clear all; clc;
figure; set(gcf, 'color','white');
% initialization params
% net ammount of activity leakage
k=0.2;
% number of neurons in the net
num_neurons = 2;
% inhibition weight
betavals = [0.0, 0.4];
% intergration step
dt = 0.1;
% standard deviation of integration noise
xi = [.0,.0];
% loop the net for MAX_EPOCHS epochs
MAX_EPOCHS = 250; % 25 s
% inputs for the 2 units
ro = zeros(2, MAX_EPOCHS);
for idx = 1:MAX_EPOCHS
    ro(:,idx) = [0.52,1-0.52];
end
% run for both beta
for bidx = 1:length(betavals)
    beta = betavals(bidx);
    % init net iterator
    net_iter = 1;
    % init activity
    x = zeros(num_neurons, MAX_EPOCHS);
    while(1)
        for idx = 1:num_neurons
            sum_ext = sum(x(:, net_iter)) - x(idx, net_iter);
            x(idx, net_iter+1) = x(idx, net_iter) + (ro(idx, net_iter) - k*x(idx, net_iter) - beta*sum_ext)*dt + xi(idx)*(sqrt(dt))^-1;
        end
        if(net_iter==MAX_EPOCHS)
            break;
        end;
        net_iter = net_iter+1;
    end;
    if(bidx==1)
        plot(x(1,:), '-r', 'LineWidth', 3); hold on;
        plot(x(2,:), '-b', 'LineWidth', 3); box off;
        % get max of non-inhibitory scenario
        MAX_ACT = max([max(x(1,:)), max(x(2,:))]);
        axis([0, MAX_EPOCHS, 0, MAX_ACT]);
    else
        plot(x(1,:), '--r', 'LineWidth', 3); hold on;
        plot(x(2,:), '--b', 'LineWidth', 3); box off;
        axis([0, MAX_EPOCHS, 0, MAX_ACT]);
    end
end
ylabel('Activities of units');
xlabel('Epochs'); legend('Unit 1', 'Unit 2');