%%%% Dynamics Controller
function tau_ff = ff_yours(th_curr, th_d_curr, th_des, th_d_des, th_dd_des, net)

    % Function to compute FF torque based on trained neural network weights
    %
    % Adaptive: use current state
    input = [wrapTo2Pi(th_curr); th_d_curr; th_dd_des];
    tau_ff = net(input); 
end