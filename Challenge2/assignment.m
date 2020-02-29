function assignment(epsilon,gamma,show_anim,live_visual)

    par = robot_set_parameters;
    par.live_visual = live_visual;
    par.epsilon = epsilon;
    par.gamma = gamma;
    par.run_type = 'learn';
    par = swingup(par);
    
    par.run_type = 'test';
    [par, ta, xa] = swingup(par);
    
    if show_anim
        animate_swingup(ta, xa, par);
    end
end



