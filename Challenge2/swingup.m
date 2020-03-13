function [par, ta, xa] = swingup(par)

    par.simtime = 10;     % Trial length
    par.simstep = 0.05;   % Simulation time step
    par.maxtorque = 1.5;  % Maximum applicable torque
    
    
    if strcmp(par.run_type, 'learn')
        %%
        % Obtain SARSA parameters
        par = get_parameters(par);
        
        Q = init_Q(par);

        % Initialize bookkeeping (for plotting only)
        ra = zeros(par.trials, 1);
        tta = zeros(par.trials, 1);
        te = 0;

        % Outer loop: trials
        for ii = 1:par.trials
            
            % Initialize a new trial
            x = swingup_initial_state();
            s = discretize_state(x, par);
            a = execute_policy(Q, s, par);
  
            % Inner loop: simulation steps
            for tt = 1:ceil(par.simtime/par.simstep)
                
                % Take the chosen action
                u = take_action(a, par);
                
                % Apply torque and obtain new state
                % x  : state (input at time t and output at time t+par.simstep)
                % u  : torque
                % te : new time
                [te, x] = body_straight([te te+par.simstep],x,u,par);
                
                % Observe next discretized state sP and reward rP
                sP = discretize_state(x, par);
                rP = observe_reward(1, sP, par); % reward does not depend on action, so random action given
                
                % Choose next action aP
                aP = execute_policy(Q, sP, par);
                
                % Learn using the  discretized state
                Q = update_Q(Q, s, a, rP, sP, aP, par);
                s = sP;
                a = aP;
      
                % Keep track of cumulative reward
                ra(ii) = ra(ii)+rP;

                % Stop trial if state is terminal
                if is_terminal(s, par)
                    break
                end
                
            end

            tta(ii) = tta(ii) + tt*par.simstep;

            % Update plot every ten trials
            if rem(ii, 10) == 0 && par.visual
                plot_Q(Q, par, ra, tta, ii);
                drawnow;
            end
            
        end
        
        close all
        f = figure('visible', 'on');
        plot_Q(Q, par, ra, tta, ii);
        str = strcat('SARSA_eps',num2str(100*par.epsilon),'_gam',num2str(100*par.gamma));
        saveas(f,strcat('Plots\',str),'epsc');
        
        % save learned Q value function
        par.Q = Q;
 
    elseif strcmp(par.run_type, 'test')
        %%
        % Obtain SARSA parameters
        par = get_parameters(par);
        
        % Read value function
        Q = par.Q;
        
        x = swingup_initial_state();
        
        ta = zeros(length(0:par.simstep:par.simtime), 1);
        xa = zeros(numel(ta), numel(x));
        te = 0;
        
        % Initialize a new trial
        s = discretize_state(x, par);
        a = execute_policy(Q, s, par);

        % Inner loop: simulation steps
        for tt = 1:ceil(par.simtime/par.simstep)
            % Take the chosen action
            TD = max(min(take_action(a, par), par.maxtorque), -par.maxtorque);

            % Simulate a time step
            [te,x] = body_straight([te te+par.simstep],x,TD,par);

            % Save trace
            ta(tt) = te;
            xa(tt, :) = x;

            s = discretize_state(x, par);
            a = execute_policy(Q, s, par);

            % Stop trial if state is terminal
            if is_terminal(s, par)
                break
            end
        end

        ta = ta(1:tt);
        xa = xa(1:tt, :);
        
    elseif strcmp(par.run_type, 'verify')
        %%
        % Get pointers to functions
        learner.get_parameters = @get_parameters;
        learner.init_Q = @init_Q;
        learner.discretize_state = @discretize_state;
        learner.execute_policy = @execute_policy;
        learner.observe_reward = @observe_reward;
        learner.is_terminal = @is_terminal;
        learner.update_Q = @update_Q;
        learner.take_action = @take_action;
        par.learner = learner;
    end
    
end

% ******************************************************************
% *** Edit below this line                                       ***
% ******************************************************************
function par = get_parameters(par)

    if isfield(par,'gamma')
    else
        par.gamma = 0.99;        % Discount rate
    end
    
    if isfield(par,'epsilon')
    else
        par.epsilon = 0.1;       % Random action rate
    end
    
    par.alpha = 0.25;        % Learning rate
    par.pos_states = 31;     % Position discretization
    par.vel_states = 31;     % Velocity discretization
    par.actions = 5;         % Action discretization
    par.trials = 2000;       % Learning trials
end

function Q = init_Q(par)
    Q = 1*ones(par.pos_states,par.vel_states,par.actions);
end

function s = discretize_state(x, par)
    x(1) = mod(x(1),2*pi);                                               % make sure the input is bounded between 0 and 2*pi
    s(1) = (par.pos_states-1)/(2*pi)*x(1) + 1;                           % using a linear mapping
    
    vel_xtrm = 5*pi;                        
    x(2) = min(max(x(2),-vel_xtrm),vel_xtrm);                            % clipping the value to -5*pi or +5*pi rad/s
    s(2) = (par.vel_states-1)/(2*vel_xtrm)*x(2) + (par.vel_states+1)/2;  % using a linear mapping
    
    s = round(s);
    
end

function u = take_action(a, par)                                         
    u = -par.maxtorque*(1+(1+par.actions)/(1-par.actions))*a + par.maxtorque*(1+par.actions)/(1-par.actions);         % using a linear mapping
    u =  min(max(u,-par.maxtorque),par.maxtorque);                                                                    % clipping the input torque to the allowable bounds
end

function r = observe_reward(a, sP, par) % actually reward does not depend on the action
    s_goal = discretize_state([pi,0], par);
    if sP==s_goal
        r = 10;
    else
        r = 0;
    end
end

function t = is_terminal(sP, par)
    s_goal = discretize_state([pi,0], par);
    if sP==s_goal
        t = 1;
    else
        t = 0;
    end
end

function a = execute_policy(Q, s, par)
    play = rand();
    if play > par.epsilon
        [~, a] = max(Q(s(1),s(2),:));
    else
        a = randi(par.actions);
    end
end

function Q = update_Q(Q, s, a, rP, sP, aP, par)
    Q(s(1),s(2),a) = Q(s(1),s(2),a) + par.alpha * (rP + par.gamma*Q(sP(1),sP(2),aP) - Q(s(1),s(2),a));
end

