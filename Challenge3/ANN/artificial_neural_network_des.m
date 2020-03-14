clc
clearvars
close all

%% GENERATE TRAINING DATA
% Input from desired state and targets from analytical equations

rot_vel = [70:80];
inputs = zeros(6,1);
targets = zeros(2,1);

for i = 1:length(rot_vel)
    [input, target] = generate_training_data(rot_vel(i));
    inputs = [inputs input];
    targets = [targets target];
end

%% TWO-LAYER FEEDFORWARD NETWORK TRAINING

net = feedforwardnet(100);
[net,tr] = train(net,inputs,targets);

%% SAVE NETWORK PARAMEETERS
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
clearvars -except net
save net_des


function [inputs,targets] = generate_training_data(rot_vel)

%% CALCULATE INPUTS: DESIRED TRAJECTORY DATA 

rp = define_robot_parameters();
sim_time = 10; % simualtion time in seconds
dt = 0.03; % time difference in seconds
t = 0:dt:sim_time;

d2r  = pi/180;             % degrees to radians
tp.w = rot_vel*d2r;            % rotational velocity rad/s
tp.rx = 1.75; tp.ry = 1.25; % ellipse radii
tp.ell_an = 45*d2r;       % angle of inclination of ellipse
tp.x0 = 0.4;  tp.y0 = 0.4;  % center of ellipse  

% Calculate desired trajectory in task space and in joint space
des = calculate_trajectory(t, tp, rp);
inputs = [des.th; des.th_d; des.th_dd];

%% TARGETS

targets = zeros(2,length(t));

for i = 1:length(t)
    targets(:,i) = ff_dyn_model_2(0, 0, inputs(1:2,i), inputs(3:4,i), inputs(5:6,i), rp);
end


end


