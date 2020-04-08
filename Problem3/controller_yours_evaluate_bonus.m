clc
clearvars
close all

% tau - torques applied to joints
% th - positions of the joints (angles)
% th_d - velocities of the joints (angular velocity)
% th_dd - acceleration of the joints
% _des - desired values (reference)
% _curr - current values (measured)
% ff_ - feedforward
% fb_ - feedback

rp = define_robot_parameters();
sim_time = 10; % simualtion time in seconds
dt = 0.03; % time difference in seconds
t = 0:dt:sim_time;

%% DESIRED TRAJECTORY DATA
d2r  = pi/180;             % degrees to radians
tp.w = 72*d2r;            % rotational velocity rad/s
tp.rx = 1.75; tp.ry = 1.25; % ellipse radii
tp.ell_an = 45*d2r;       % angle of inclination of ellipse
tp.x0 = 0.4;  tp.y0 = 0.4;  % center of ellipse  

rot_vel = 70:1:80;
RMSE_yours.x = zeros(1,length(rot_vel));
RMSE_yours.th = zeros(1,length(rot_vel));
RMSE_DYN1.x = zeros(1,length(rot_vel));
RMSE_DYN1.th = zeros(1,length(rot_vel));

Kp = [500; 500];
Kd = [50; 50];

% Your Code
folder = fileparts(which(mfilename));
addpath(genpath(folder));
load ANN/net.mat
initial_offset_vector = zeros(2,length(rot_vel)); % array with initial offsets for both links

%% SIMULATE ROBOT

for iter = 1:length(rot_vel)
    tp.w = rot_vel(iter)*d2r;

    % Calculate desired trajectory in task space and in joint space
    des = calculate_trajectory(t, tp, rp);

    % Choose offset randomly between -30deg and 30deg for both links
    initial_offset = [randi([-30,30]); randi([-30,30])]; 
    initial_offset_vector(:,iter) = initial_offset;
    
    th_0 = des.th(:,1) - initial_offset*pi/180;
    th_d_0 = des.th_d(:,1);
    
    % Data-driven controller
    curr_yours = simulate_robot(t, dt, th_0, th_d_0, des, rp, ...
        @(th_curr, th_d_curr, th_des, th_d_des, th_dd_des) ff_yours(th_curr, th_d_curr, th_des, th_d_des, th_dd_des, net), ...
        @(th_curr, th_d_curr, th_des, th_d_des) fb_pd(th_curr, th_d_curr, th_des, th_d_des, Kp, Kd));

    [RMSE_yours.x(iter), RMSE_yours.th(iter)] = analyze_performance(t, curr_yours, des, false);
    
    % Adaptive model-reference controller
    curr_dyn1 = simulate_robot(t, dt, th_0, th_d_0, des, rp, ...
        @(th_curr, th_d_curr, th_des, th_d_des, th_dd_des) ff_dyn_model_1(th_curr, th_d_curr, th_des, th_d_des, th_dd_des, rp), ...
        @(th_curr, th_d_curr, th_des, th_d_des) fb_pd(th_curr, th_d_curr, th_des, th_d_des, Kp, Kd));

    [RMSE_DYN1.x(iter), RMSE_DYN1.th(iter)] = analyze_performance(t, curr_dyn1, des, false);
    
end

%% PLOT COMPARISON

f1 = figure('visible', 'on','Position', [400 400 750 300]);
plot(rot_vel, RMSE_yours.x','-s','LineWidth',1.5,'MarkerFaceColor',[0    0.4470    0.7410]);
hold on
plot(rot_vel, RMSE_DYN1.x','-s','LineWidth',1.5,'MarkerFaceColor',[0.8500    0.3250    0.0980]);
xlim([70 80]);
ylim([0.05 0.2]);
pxy = get(gca, 'Position');
px = pxy([1 3]);
py = pxy([2 4]);
lx = xlim;
ly = ylim;
dlx = diff(lx);
dly = diff(ly);
% Add initial offset values as annotations
for i = 1:length(rot_vel)
    if any([2,4,6,7,8,10,11] == i) 
        Q1 = [((rot_vel(i)-lx(1))/12.9+px(1)).*[1., 1.], (RMSE_DYN1.x(i)-ly(1))/0.18+py(1)+0.07, (RMSE_DYN1.x(i)-ly(1))/0.18+py(1)];
    else
        Q1 = [((rot_vel(i)-lx(1))/12.9+px(1)).*[1., 1.], (RMSE_DYN1.x(i)-ly(1))/0.18+py(1)-0.07, (RMSE_DYN1.x(i)-ly(1))/0.18+py(1)];
    end
    annotation('textarrow', Q1(1:2), Q1(3:4), 'String',strcat('',num2str(initial_offset_vector(1,i)),'°,  ',num2str(initial_offset_vector(2,i)),'°'),'TextBackgroundColor','white','TextEdgeColor',[0.4 0.4 0.4]);
end
ylabel('RMSE x [m]');
legend('yours','DYN1');
grid on
set(findall(gcf,'-property','FontSize'),'FontSize',15);
annotation('textbox', [0.735 0.658 0.165 0.095], 'String','Initial Offset','BackgroundColor','white','EdgeColor',[0.4 0.4 0.4],'HorizontalAlignment','right','FontSize',13)%,'FitBoxToText','on');
annotation('textbox', [0.74 0.67 0.057 0.072], 'String','\theta_{1}, \theta_{2}','BackgroundColor','white','EdgeColor',[0.4 0.4 0.4],'FontSize',13)%,'FitBoxToText','on');
set(legend,'FontName','Helvetica','Location','Northeast'); 
saveas(f1,'evaluation_bonus_x','epsc');


f2 = figure('visible', 'on','Position', [400 400 750 300]);
plot(rot_vel, RMSE_yours.th','-s','LineWidth',1.5,'MarkerFaceColor',[0    0.4470    0.7410]);
hold on
plot(rot_vel, RMSE_DYN1.th','-s','LineWidth',1.5,'MarkerFaceColor',[0.8500    0.3250    0.0980]);
xlim([70 80]);
xlabel('rot vel [deg/s]');
ylim([0.06 0.12]);
pxy = get(gca, 'Position');
px = pxy([1 3]);
py = pxy([2 4]);
lx = xlim;
ly = ylim;
dlx = diff(lx);
dly = diff(ly);
% Add initial offset values as annotations
for i = 1:length(rot_vel)
    if any([1,2,4,5,6,7,8,11] == i) 
        Q1 = [((rot_vel(i)-lx(1))/12.9+px(1)).*[1., 1.], (RMSE_yours.th(i)-ly(1))/0.073+py(1)+0.09, (RMSE_yours.th(i)-ly(1))/0.073+py(1)+0.03];
    else
        Q1 = [((rot_vel(i)-lx(1))/12.9+px(1)).*[1., 1.], (RMSE_yours.th(i)-ly(1))/0.072+py(1)-0.07, (RMSE_yours.th(i)-ly(1))/0.072+py(1)];
    end
    annotation('textarrow', Q1(1:2), Q1(3:4), 'String',strcat('',num2str(initial_offset_vector(1,i)),'°,  ',num2str(initial_offset_vector(2,i)),'°'),'TextBackgroundColor','white','TextEdgeColor',[0.4 0.4 0.4]);
end
ylabel('RMSE \theta [rad]');
yticks([0.06 0.08 0.1 0.12])
legend('yours','DYN1');
grid on
set(findall(gcf,'-property','FontSize'),'FontSize',15);
annotation('textbox', [0.735 0.658 0.165 0.095], 'String','Initial Offset','BackgroundColor','white','EdgeColor',[0.4 0.4 0.4],'HorizontalAlignment','right','FontSize',13)%,'FitBoxToText','on');
annotation('textbox', [0.74 0.67 0.057 0.072], 'String','\theta_{1}, \theta_{2}','BackgroundColor','white','EdgeColor',[0.4 0.4 0.4],'FontSize',13)%,'FitBoxToText','on');
set(legend,'FontName','Helvetica','Location','Northeast'); 
xlabel('rot vel [deg/s]');
saveas(f2,'evaluation_bonus_th','epsc');

fprintf('mean RMSE x\n');
fprintf('yours %f\n', mean(RMSE_yours.x));
fprintf('DYN1 %f\n', mean(RMSE_DYN1.x));
