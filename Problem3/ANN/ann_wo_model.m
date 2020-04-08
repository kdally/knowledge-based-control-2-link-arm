clc
clearvars
close all
tmp = matlab.desktop.editor.getActive;
parts = strsplit(fileparts(tmp.Filename), '/');
parent_path = strjoin(parts(1:end-1), '/');
cd(parent_path);
addpath(fileparts(tmp.Filename)); % make sure all the functions are in the path

%% GENERATE TRAINING DATA
% Input from desired state and targets from analytical equations

rot_vel = [70:1:72];

% For bonus point:　various initial offset value for both links, for +-30deg
% initial_offset_1 = [-30:10:30 0.1*180/pi];
% initial_offset_2 = [-30:10:30 0.2*180/pi];
initial_offset_1 = [0.1*180/pi 0.*180/pi -0.1*180/pi];
initial_offset_2 = [0.2*180/pi 0.*180/pi -0.2*180/pi];

% Initialize input and target arrays with the right size
y = zeros(6,1);
y_des = zeros(6,1);
u = zeros(2,1);

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
%             y_this = [wrapTo2Pi(curr.th); curr.th_d; curr.th_dd];
%             y_des_this = [wrapTo2Pi(des.th); des.th_d; des.th_dd];
            y_this = [wrapTo2Pi(curr.th); curr.th_d; curr.th_dd];
            y_des_this = [wrapTo2Pi(des.th); des.th_d; des.th_dd];            

            % Record the model-based controller output
            u_this = curr.tau_ff;
            
            % Take into account that the controller works with values at
            % (iter-1) in simulate_robot
            y = [y y_this(1:6,1:end-1)];
            y_des = [y_des y_des_this(1:6,1:end-1)];
            u = [u u_this(:,2:end)];
%             y = [y y_this(1:4,1:end)];
%             y_des = [y_des y_des_this(1:4,1:end)];
%             u = [u u_this(:,1:end)];
        end 
    end
end

% Remove the column of zeros due to the initialization
y = con2seq(y(:,2:end));
y_des = con2seq(y_des(:,2:end));
u = con2seq(u(:,2:end));

disp("Done preparing training data")


%%

d1 = [1:2];
d2 = [1:2];
S1 = 30;
narx_net.divideFcn = '';
narx_net = narxnet(d1,d2,S1);
% narx_net.trainParam.min_grad = 1e-10;
[p,Pi,Ai,t] = preparets(narx_net,u,{},y);
% Give some initial random weights to network
narx_net.trainFcn = 'trainbr';
narx_net.trainParam.epochs = 2000;
narx_net = init(narx_net);
narx_net = train(narx_net,p,t,Pi);
narx_net_closed = closeloop(narx_net);
view(narx_net_closed)

%% MRAC NET
load net

mrac_net = feedforwardnet([30 2 S1]);
mrac_net.layerConnect = [0 1 0 1;       % make the first hidden layer receive feedback from layer 2 and 4
                         1 0 0 0;       % make layer 2 receive feedfoward from layer 1
                         0 1 0 1;       % make layer three receive feedforward from layer 2 and feedback from layer 4
                         0 0 1 0];      % make layer 4 receive feedforward from layer 3 
mrac_net.outputs{4}.feedbackMode = 'closed';
mrac_net.layers{2}.transferFcn = 'purelin';
mrac_net.layerWeights{3,4}.delays = 1:2;
mrac_net.layerWeights{3,2}.delays = 1:2;
mrac_net.layerWeights{3,2}.learn = 0;  % no learning for weights already trained 
mrac_net.layerWeights{3,4}.learn = 0;
mrac_net.layerWeights{4,3}.learn = 0;

% CHECK
% mrac_net.inputWeights{1}.learn = 0; % CHECK
% mrac_net.layerWeights{2,1}.learn = 0; % CHECK
mrac_net.biases{1}.learn = 0;         % CHECK
mrac_net.biases{2}.learn = 0;         % CHECK


mrac_net.biases{3}.learn = 0;          % no learning for biases already trained
mrac_net.biases{4}.learn = 0;
mrac_net.name = 'Model Reference Adaptive Control Network';
mrac_net.layerWeights{1,2}.delays = 1:1;
mrac_net.layerWeights{1,4}.delays = 1:3;
mrac_net.inputWeights{1}.delays = 1:1;

mrac_net = configure(mrac_net,y_des,y);


mrac_net = init(mrac_net);

mrac_net.b{1} = net.b{1};
mrac_net.b{2} = net.b{2};
mrac_net.IW{1} = net.IW{1};
mrac_net.LW{2,1} = net.LW{2,1};

mrac_net.LW{3,2} = narx_net_closed.IW{1};
mrac_net.LW{3,4} = narx_net_closed.LW{1,2};
mrac_net.b{3} = narx_net_closed.b{1};
mrac_net.LW{4,3} = narx_net_closed.LW{2,1};
mrac_net.b{4} = narx_net_closed.b{2};


mrac_net.LW{2,1} = zeros(size(mrac_net.LW{2,1}));
mrac_net.b{2} = [0; 0];
% mrac_net.LW{1,4} = zeros(size(mrac_net.LW{1,4}));
% mrac_net.LW{1,2} = zeros(size(mrac_net.LW{1,2}));
% mrac_net.b{1} = zeros(size(mrac_net.b{1}));

% view(mrac_net)

% mrac_net.divideFcn = '';
mrac_net.trainFcn = 'trainbr';
[x_tot,xi_tot,ai_tot,t_tot] = preparets(mrac_net,y_des,{},y);
mrac_net.trainParam.epochs = 500;
mrac_net.trainParam.min_grad = 1e-10;
mrac_net.trainParam.mu_max = 1e15;
[mrac_net,tr] = train(mrac_net,x_tot,t_tot,xi_tot,ai_tot);


%% export to mrac_cont_net

mrac_cont_net = feedforwardnet([70 2 S1 6]);
mrac_cont_net.layerConnect = [0 1 0 1 0;       % make the first hidden layer receive feedback from layer 2 and 4
                         1 0 0 0 0;       % make layer 2 receive feedfoward from layer 1
                         0 1 0 1 0;       % make layer three receive feedforward from layer 2 and feedback from layer 4
                         0 0 1 0 0;
                         0 1 0 0 0];      % make layer 4 receive feedforward from layer 3 
mrac_cont_net = configure(mrac_cont_net,y_des,u);
mrac_cont_net.layers{2}.transferFcn = 'purelin';
mrac_cont_net.layerWeights{3,4}.delays = 1:2;
mrac_cont_net.layerWeights{3,2}.delays = 1:2;

mrac_cont_net.layerWeights{1,2}.delays = 1:2;
mrac_cont_net.layerWeights{1,4}.delays = 1:3;
mrac_cont_net.inputWeights{1}.delays = 1:2;


view(mrac_cont_net)
mrac_cont_net.LW{3,2} = narx_net_closed.IW{1};
mrac_cont_net.LW{3,4} = narx_net_closed.LW{1,2};
mrac_cont_net.b{3} = narx_net_closed.b{1};
mrac_cont_net.LW{4,3} = narx_net_closed.LW{2,1};
mrac_cont_net.b{4} = narx_net_closed.b{2};
mrac_cont_net.IW{1} = mrac_net.IW{1};
mrac_cont_net.b{1} = mrac_net.b{1};
mrac_cont_net.LW{1,2} = mrac_net.LW{1,2};
mrac_cont_net.LW{1,4} = mrac_net.LW{1,4};

mrac_cont_net.LW{2,1} = mrac_net.LW{2,1};
mrac_cont_net.b{2} = mrac_net.b{2};
mrac_cont_net.LW{5,2} = ones(size(mrac_cont_net.LW{5,2}));
mrac_cont_net.b{2} = [0; 0];

view(mrac_cont_net)
%%
y_des_this2 = con2seq(y_des_this(1:2,:));
testout = mrac_net(y_des_this2);
testout = cell2mat(testout);
figure
plot([y_des_this(1:1,:)' testout(1:1,:)'])


%% SAVE TRAINED NETWORK

tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
parts = strsplit(fileparts(tmp.Filename), '/');
parent_path = strjoin(parts(1:end-1), '/');
% clearvars -except net parent_path
save mrac_cont_net         % save the network in the sub-directory
cd(parent_path); % go back to the main directory