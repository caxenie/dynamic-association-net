%% Dynamic association model to learn correlations between paired modalities
%  samples acquired from different sensors
%  current scenario is using only 2 input modalities

%% SIMULATION PARAMS
clear all; clc; close all; pause(2);
format long g;
fprintf('Simulation started ... \n');

% number of neurons in each input layer
INPUT_NEURONS = 40;
% number of neurons in association layer
ASSOCIATION_NEURONS = 40;
% enable dynamic visualization of encoding process
DYN_VISUAL = 1;
% enable verbose 
VERBOSE = 0;

% learning rate for feedforward propagation of sensory afferents
TAU_FF = 0.01;
% learning rate for feedback propagation of association projections
TAU_FB = 0.1;
% learning rate for weight adaptation
ETA = 0.001;
% choose the stability point of the network
STABLE_POINT = 1e-10;

% inhibition in association layer
INH_ASSOC = 2;

% inhibition in input layers
% I = 2     --> strong inhibition
% I = (1,2) --> less inhibition
% I = 1     --> no inhibition
% I < 1     --> activation collpases to resting state
INH_IN = 1.2;

%% INITIALIZATION
% define the network's layers as evolving in time
input_layer1 = struct('lsize', INPUT_NEURONS, ...
    'W', rand(INPUT_NEURONS, ASSOCIATION_NEURONS), ...
    'A', zeros(1, INPUT_NEURONS));

input_layer2 = struct('lsize', INPUT_NEURONS, ...
    'W', rand(INPUT_NEURONS, ASSOCIATION_NEURONS), ...
    'A', zeros(1, INPUT_NEURONS));

assoc_layer = struct('lsize', ASSOCIATION_NEURONS, ...
    'A', zeros(1, ASSOCIATION_NEURONS));

% changes in activity for each neuron in each layer
delta_act_assoc  = ones(1, assoc_layer(1).lsize)/ assoc_layer(1).lsize;
old_delta_act_assoc  = zeros(1, assoc_layer(1).lsize);

% network iterator
net_iter = 0;
% epochs iterator
net_epoch = 1;

% prepare the sensory dataset
sensory_data = sensory_data_setup('robot_data_jras_paper', 'tracker_data_jras_paper');
% run the network for the entire robot sensory dataset
MAX_EPOCHS = length(sensory_data.timeunits);
if (VERBOSE==1), fprintf('Loaded sensory dataset with %d samples ... \n\n', MAX_EPOCHS); end
% visualization init
figure; set(gcf, 'color', 'w');

%--------------------------
% timing the simulation time
t_start_simulation = tic;
t_exe_hist = zeros(MAX_EPOCHS, 1);
%--------------------------

%% NETWORK DYNAMICS
% loop throught training epochs
while(1)
    fprintf('------------------- epoch %d --------------------- \n', net_epoch);
    %% INPUT DATA
    
    % visualize encoding process
    if(DYN_VISUAL==1)
        % input
        subplot(2,3,1);
        acth1 = plot(input_layer1.A, '-r', 'LineWidth', 2); box off;
        xlabel('neuron index'); ylabel('activation input layer 1');
        subplot(2,3,2);
        acth2 = plot(input_layer2.A, '-b','LineWidth', 2); box off;
        xlabel('neuron index'); ylabel('activation input layer 2');
        subplot(2,3,3);
        acth3 = plot(assoc_layer.A, '-k','LineWidth', 2); box off;
        xlabel('neuron index'); ylabel('activation association layer');
        % weights
        subplot(2,3,4);
        vis_data1 = input_layer1.W(1:INPUT_NEURONS, 1:ASSOCIATION_NEURONS)';
        acth4 = pcolor(vis_data1); %colorbar;
        box off; grid off; axis xy; xlabel('input layer 1 - neuron index'); ylabel('association layer - neuron index');
        subplot(2,3,5);
        vis_data2 = input_layer2.W(1:INPUT_NEURONS, 1:ASSOCIATION_NEURONS)';
        acth5 = pcolor(vis_data2); %colorbar;
        box off; grid off; axis xy; xlabel('input layer 2 - neuron index'); ylabel('association layer - neuron index');
        
        % motion diagram 
        subplot(2,3,6);
        acth7 = plot(sensory_data.pose.cam(2, :), sensory_data.pose.cam(1, :),'-g', 'LineWidth', 2); hold on;
        acth6 = plot(sensory_data.pose.cam(2, net_epoch), sensory_data.pose.cam(1, net_epoch),'o', 'LineWidth', 3, 'MarkerEdgeColor','k',...
                                                                                    'MarkerFaceColor','w',...
                                                                                    'MarkerSize',20);
        box off; grid off; axis([min(sensory_data.pose.cam(2, :))-0.125, ...
                                 max(sensory_data.pose.cam(2, :))+0.125,...
                                 min(sensory_data.pose.cam(1, :))-0.125,...
                                 max(sensory_data.pose.cam(1, :))+0.125]); 
        xlabel('distance (m)'); ylabel('Robot motion'); 
        
        % refresh visualization
        set(acth1, 'YDataSource', 'input_layer1.A');
        set(acth2, 'YDataSource', 'input_layer2.A');
        set(acth3, 'YDataSource', 'assoc_layer.A');
        set(acth4, 'CData', vis_data1);
        set(acth5, 'CData', vis_data2);
        set(acth6, 'YDataSource', 'sensory_data.pose.cam(1, net_epoch)');
        set(acth6, 'XDataSource', 'sensory_data.pose.cam(2, net_epoch)');
        refreshdata(acth1, 'caller');
        refreshdata(acth2, 'caller');
        refreshdata(acth3, 'caller');
        refreshdata(acth4, 'caller');
        refreshdata(acth5, 'caller');
        refreshdata(acth6, 'caller');       
        drawnow;
    end
    
    % reset history
    old_delta_act_assoc = zeros(1, assoc_layer(1).lsize);
    % get one sample from robot data and encode it in population activity
    input_layer1.A = population_encoder(sensory_data.heading.gyro(net_epoch)*pi/180, INPUT_NEURONS);
    input_layer2.A = population_encoder(sensory_data.heading.odometry(net_epoch)*pi/180, INPUT_NEURONS);
    
    % input data is normalized in [0,1], so the sum of all neuron activities
    % should be summing up to 1
    input_layer1.A = input_layer1.A./sum(input_layer1.A);
    input_layer2.A = input_layer2.A./sum(input_layer2.A);    

    %--------------------------
    % timing the settling time
    t_start_settling = tic;
    %--------------------------

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
        if((sum((old_delta_act_assoc - delta_act_assoc).^2))<=STABLE_POINT)
            if (VERBOSE==1), fprintf('network has settled after %d iterations --> ', net_iter); end;
            net_iter = 0;
            break;
        end
        
        % update history
        old_delta_act_assoc = delta_act_assoc;
        net_iter = net_iter + 1;
                
    end % end of an epoch
    
    %--------------------------
    % timing the settling time 
    t_elapsed_settling = toc(t_start_settling); 
    t_exe_hist(net_epoch) = t_elapsed_settling;
    fprintf(' %f s \n', t_elapsed_settling);
    %--------------------------
    
    % check if end of simulation
    if(net_epoch==MAX_EPOCHS)
        if (VERBOSE==1), fprintf('network has learned in %d epochs \n', net_epoch); end;
        break;
    end
    % epoch increment
    net_epoch = net_epoch + 1;
end


%-------------------------------
% end timing the simulation time
t_elapsed_simulation = toc(t_start_simulation);
%-------------------------------

if (VERBOSE==1), printf('\n Mean settling time is %d s\n', mean(t_exe_hist)); end;
if (VERBOSE==1), printf('\n Total execution time is %d mins \n', t_elapsed_simulation/60); end;

%% VISUALIZATION
% weights after learning
figure; set(gcf, 'color', 'white');
subplot(1,2,1);
imagesc(input_layer1.W(1:INPUT_NEURONS, 1:ASSOCIATION_NEURONS)'); colormap; colorbar;
box off; grid off; axis xy; xlabel('input layer 1'); ylabel('association layer');
subplot(1,2,2);
imagesc(input_layer2.W(1:INPUT_NEURONS, 1:ASSOCIATION_NEURONS)'); colormap; colorbar;
box off; grid off; axis xy; xlabel('input layer 2'); ylabel('association layer');

