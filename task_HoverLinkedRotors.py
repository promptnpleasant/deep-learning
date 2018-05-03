import numpy as np
from physics_sim import PhysicsSim

class task_HoverLinkedRotors():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 3
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.ceiling = self.target_pos[2]+20
        self.floor = 1

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward of -20 for remaining time if hit ground or ceiling
        if (self.sim.pose[2] <= self.floor) or (self.sim.pose[2] >= self.ceiling):
            beyond_bounds_dist = self.floor - self.sim.pose[2]
            if beyond_bounds_dist < 0:
                beyond_bounds_dist = self.sim.pose[2] - self.ceiling
            print('beyond_bounds_dist',beyond_bounds_dist)
            time_remaining = self.sim.runtime-self.sim.time
            print()
            print('time_remaining', time_remaining)
            ticks_remaining = time_remaining / self.sim.dt
            print('ticks_remaining', ticks_remaining)
            print('action_repeat', float(self.action_repeat))
            reward_opportunities = ticks_remaining / float(self.action_repeat)
            print('reward_opportunities', reward_opportunities)
            reward = -2.0 * reward_opportunities + -10 * beyond_bounds_dist #make it not worth it to fail now (-2 * remaining ticks)
            print('out-of-bounds reward=', reward)
        elif abs(self.sim.pose[2] - self.target_pos[2]) < 1:
            reward = 1.0-1.0*abs(self.sim.pose[2] - self.target_pos[2])  #0 to 1
        else:
            reward = max(1.0-1.0*abs(self.sim.pose[2] - self.target_pos[2]), -19) #0 to -20)
        return reward
        
        # z=target+20 => -2 * all remaining time steps and simulation ends here
        #   ...
        # z=target+10 =>-1
        #   ...
        # z=target+1  => 0
        # z=target    => 1
        # z=target-1  => 0
        #   ...
        # z=target-10 => -1
        #
        # z=0 or z=target-20 => -2 * all remaining time steps and simulation ends here
        
        

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds*4) # update the sim pose and velocities
            #if (reward > -10 * self.action_repeat):
            #   reward += self.get_reward()
            pose_all.append([self.sim.pose[2], self.sim.v[2], self.sim.linear_accel[2]])
            
        next_state = np.concatenate(pose_all)
        reward += self.get_reward()
        if (self.sim.pose[2] <= self.floor) or (self.sim.pose[2] >= self.ceiling):
            done = True    #stop the simulation if outside box
            pass
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = [self.sim.pose[2], self.sim.v[2], self.sim.linear_accel[2]] * self.action_repeat 
        return state