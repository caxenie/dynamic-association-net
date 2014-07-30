%% Dynamic association model to learn correlations between paired modalities
%  samples acquired from different sensors
%  current scenario is using only 2 input modalities

%% SIMULATION PARAMS
clear all; clc; close all; pause(2);

fprintf('Simulation started ... \n');

% number of neurons in each input layer
INPUT_NEURONS = 40;
% number of neurons in association layer
ASSOCIATION_NEURONS = 80;

% learning rate for feedforward propagation of sensory afferents
TAU_FF = 0.001;
% learning rate for feedback propagation of association projections
TAU_FB = 2;
% learning rate for weight adaptation
ETA = 0.002;

% inhibition in association layer
INH_ASSOC = 2;

% inhibition in input layers
% I = 2     --> strong inhibition
% I = (1,2) --> less inhibition
% I = 1     --> no inhibition
% I < 1     --> activation collpases to resting state
INH_IN = 1.0;

%% INITIALIZATION
% define the network's layers as evolving in time
input_layer1 = struct('lsize', INPUT_NEURONS, ...
    'W', randn(INPUT_NEURONS, ASSOCIATION_NEURONS), ...
    'A', zeros(1, INPUT_NEURONS));

input_layer2 = struct('lsize', INPUT_NEURONS, ...
    'W', randn(INPUT_NEURONS, ASSOCIATION_NEURONS), ...
    'A', zeros(1, INPUT_NEURONS));

assoc_layer = struct('lsize', ASSOCIATION_NEURONS, ...
    'A', zeros(1, ASSOCIATION_NEURONS));

% changes in activity for each neuron in each layer
delta_act_assoc  = zeros(1, assoc_layer(1).lsize);
old_delta_act_assoc  = zeros(1, assoc_layer(1).lsize);

% network iterator
net_iter = 0;
% epochs iterator
net_epoch = 1;

% prepare the sensory dataset
sensory_data = sensory_data_setup('robot_data_jras_paper', 'tracker_data_jras_paper');
% run the network for the entire robot sensory dataset
MAX_EPOCHS = length(sensory_data.timeunits);
fprintf('Loaded sensory dataset with %d samples ... \n\n', MAX_EPOCHS);

%% NETWORK DYNAMICS
% loop throught training epochs
while(1)    
    fprintf('------------------- epoch %d --------------------- \n', net_epoch);
    %% INPUT DATA
    % reset history 
    old_delta_act_assoc = zeros(1, assoc_layer(1).lsize);
    % get one sample from robot data and encode it in population activity
    input_layer1.A = population_encoder(sensory_data.heading.gyro(net_epoch), INPUT_NEURONS);
    input_layer2.A = population_encoder(sensory_data.heading.odometry(net_epoch), INPUT_NEURONS);
    
    % input data is normalized in [0,1], so the sum of all neuron activities
    % should be summing up to 1
     input_layer1.A = input_layer1.A./sum(input_layer1.A);
     input_layer2.A = input_layer2.A./sum(input_layer2.A);
   
    % iterate the network until it settles
    while(1)
        % -------------------------------------------------------------------------------------------------------------------------
        % FEEDFORWARD PATHWAY
        
        % project the sensory afferent activity from the input layers onto
        % association layer, after presenting paired sample from each sensor
        delta_act_assoc = input_layer1.A*input_layer1.W + input_layer2.A*input_layer2.W;
        
        % update neural activity in the association layer
        assoc_layer.A = assoc_layer.A + TAU_FF*delta_act_assoc;
        
        % after updating association neurons inhibit eachother using squared
        % normalization --> when run repeadetly this will approximate a WTA
        assoc_layer.A = assoc_layer.A.^INH_ASSOC;
        assoc_layer.A = assoc_layer.A./sum(assoc_layer.A);
        
        % -------------------------------------------------------------------------------------------------------------------------
        % FEEDBACK PATHWAY
        
        % update each input layer activity
        input_layer1.A = input_layer1.A + TAU_FB*input_layer1.A.*(assoc_layer.A*input_layer1.W');
        input_layer2.A = input_layer2.A + TAU_FB*input_layer2.A.*(assoc_layer.A*input_layer2.W');
        
        % apply inhibition and normalize the activity in the input layers
        input_layer1.A = (input_layer1.A).^INH_IN;
        input_layer1.A = input_layer1.A./(sum(input_layer1.A));
        input_layer2.A = (input_layer2.A).^INH_IN;
        input_layer2.A = input_layer2.A./(sum(input_layer2.A));
        
        % -------------------------------------------------------------------------------------------------------------------------
        % WEIGHT ADAPTATION
        
        % Hebbian learning - strengthen the simulatanously active input and
        % association neurons (co-activated) and decrease the others
        
        input_layer1.W = input_layer1.W + ETA*...
            (input_layer1.A'*assoc_layer.A.*(1-input_layer1.W) - ...
            0.5*(((1-input_layer1.A')*assoc_layer.A + input_layer1.A'*(1 - assoc_layer.A)).*input_layer1.W));
        
        input_layer2.W = input_layer2.W + ETA*...
            (input_layer2.A'*assoc_layer.A.*(1-input_layer2.W) - ...
            0.5*(((1-input_layer2.A')*assoc_layer.A + input_layer2.A'*(1 - assoc_layer.A)).*input_layer2.W));
        
        % weights normalization
        input_layer1.W = (input_layer1.W - min(input_layer1.W(:)))/(max(input_layer1.W(:)) - min(input_layer1.W(:)));
        input_layer2.W = (input_layer2.W - min(input_layer2.W(:)))/(max(input_layer2.W(:)) - min(input_layer2.W(:)));
        
        % -------------------------------------------------------------------------------------------------------------------------
        
        % stop condition
        if((sum((old_delta_act_assoc - delta_act_assoc).^2))<=1e-12)
            fprintf('network has settled after %d iterations \n', net_iter);
            net_iter = 0;
            break;
        end
        % update history
        old_delta_act_assoc = delta_act_assoc;
        net_iter = net_iter + 1;
    end % end of an epoch
    % check if end of simulation
    if(net_epoch==MAX_EPOCHS)
        fprintf('network has learned in %d epochs \n', net_epoch);
        break;
    end
    % epoch increment
    net_epoch = net_epoch + 1;
end

%% VISUALIZATION
% input
figure; set(gcf, 'color', 'white');
subplot(1,3,1);
plot(input_layer1.A); xlabel('Neuron index'); ylabel('Activation input layer 1');
box off; grid off;
subplot(1,3,2);
plot(input_layer2.A); xlabel('Neuron index'); ylabel('Activation input layer 2');
box off; grid off;
subplot(1,3,3);
plot(assoc_layer.A); xlabel('Neuron index'); ylabel('Activation association layer');
box off; grid off;
% weights
figure; set(gcf, 'color', 'white');
subplot(1,2,1);
imagesc(input_layer1(end).W(1:INPUT_NEURONS, 1:ASSOCIATION_NEURONS)'); colormap; colorbar;
box off; grid off; axis xy; xlabel('input layer 1'); ylabel('association layer');
subplot(1,2,2);
imagesc(input_layer2(end).W(1:INPUT_NEURONS, 1:ASSOCIATION_NEURONS)'); colormap; colorbar;
box off; grid off; axis xy; xlabel('input layer 2'); ylabel('association layer');

