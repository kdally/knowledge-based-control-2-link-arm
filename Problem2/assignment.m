function assignment(epsilon,gamma,visual)
 
    par = robot_set_parameters;
    
    if nargin == 0  
        par.visual = true;
    elseif nargin == 2
        par.gamma = gamma;
        par.epsilon = epsilon;
    else
        par.gamma = gamma;
        par.epsilon = epsilon;
        par.visual = visual;
    end
    
    par.run_type = 'learn';
    par = swingup(par);
    
    par.run_type = 'test';
    [par, ta, xa] = swingup(par);
    
    if par.visual
        animate_swingup(ta, xa, par);
    end
end



