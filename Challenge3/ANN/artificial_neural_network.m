clc
clearvars
close all
% MAKE SURE TO CD THE PARENT FOLDER OF ANN

% For MacOS, use:
%
% tmp = matlab.desktop.editor.getActive;
% parts = strsplit(fileparts(tmp.Filename), '/');
% parent_path = strjoin(parts(1:end-1), '/');
% cd(parent_path); % go back to the main directory

%% GENERATE TRAINING DATA
% Input from desired state and targets from analytical equations

rot_vel = [70:1:80];

% For bonus point:　various initial offset value for both links, for +-30deg
initial_offset_1 = [-30:10:30 0.1*180/pi];
initial_offset_2 = [-30:10:30 0.2*180/pi];

% Initialize input and target arrays with the right size
inputs = zeros(6,1);
targets = zeros(2,1);

% Runtime: 15sec
for i = 1:length(rot_vel)
    for j = 1:length(initial_offset_1)
        for k = 1:length(initial_offset_2)
            
            % Convert to radians
            initial_offset = [initial_offset_1(j)*pi/180; initial_offset_2(k)*pi/180];
            
            % Capture data from controller_1 for given rotational velocity
            % and initial offset
            [curr, des] = controller_1_func(rot_vel(i), initial_offset);

            % Record state and wrap around theta current
            input = [wrapTo2Pi(curr.th); curr.th_d; des.th_dd];
            
            % Record the model-based controller output
            target = curr.tau_ff;
            
            % Take into account that the controller works with values at
            % (iter-1) in simulate_robot
            inputs = [inputs input(:,1:end-1)];
            targets = [targets target(:,2:end)];
        end 
    end
end

% Remove the column of zeros due to the initialization
inputs = inputs(:,2:end);
targets = targets(:,2:end);

disp("Done preparing training data")

%% TWO-LAYER FEEDFORWARD NETWORK TRAINING

% Create ANN with hidden layer of 25 neurons
net = feedforwardnet(25);

% Remove duplicates and rondomize dataset
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand';

% Specify proportion of data for training, validation and testing
net.divideParam.valRatio = 0.75;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Training parameters
net.trainFcn = 'trainlm';       % Levenberg-Marquardt method for training
net.trainParam.epochs = 2000;   % Maximum number of iterations
net.performFcn = 'mse';         % Performance function

% User preference
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = true;

% Give some initial random weights to network
net = init(net);

% Train until 2000 epochs reached or mse increases
[net,tr] = train(net,inputs,targets);

%% SAVE TRAINED NETWORK

tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
parts = strsplit(fileparts(tmp.Filename), '/');
parent_path = strjoin(parts(1:end-1), '/');
clearvars -except net parent_path
save net         % save the network in the sub-directory
cd(parent_path); % go back to the main directory

