clc
clearvars
close all

%% GENERATE TRAINING DATA
% Input from desired state and targets from analytical equations

rot_vel = [70:1:80];
initial_offset_1 = 0.1*180/pi;
initial_offset_2 = 0.2*180/pi;
% initial_offset_1 = [-30:5:30];
% initial_offset_2 = [-30:5:30];
inputs = zeros(6,1);
targets = zeros(2,1);

iter = 0;
for i = 1:length(rot_vel)
    for j = 1:length(initial_offset_1)
        for k = 1:length(initial_offset_2)
            
            initial_offset = [initial_offset_1(j)*pi/180; initial_offset_2(k)*pi/180];
            [curr, des] = controller_1_func(rot_vel(i), initial_offset);

            input = [curr.th; curr.th_d; des.th_dd];
            target = curr.tau_ff;
            inputs = [inputs input(:,1:end-1)];
            targets = [targets target(:,2:end)];
        
        end
        iter = iter+1;
    end
end

inputs = inputs(:,2:end);
targets = targets(:,2:end);

%% TWO-LAYER FEEDFORWARD NETWORK TRAINING

trainFcn = 'trainlm';
hiddenLayerSize = 100;
net = fitnet(hiddenLayerSize,trainFcn);

net.divideParam.valRatio = 0.75;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
net.trainParam.epochs = 1000;
net.performFcn = 'mse';

net.trainParam.showWindow = false;
net.trainParam.showCommandLine = true;
[net,tr] = train(net,inputs,targets);

% SAVE NETWORK PARAMEETERS

tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
parts = strsplit(fileparts(tmp.Filename), '/');
parent_path = strjoin(parts(1:end-1), '/');
clearvars -except net parent_path
save net
cd(parent_path);

