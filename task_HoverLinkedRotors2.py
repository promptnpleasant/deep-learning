import numpy as np
from physics_sim import PhysicsSim

class task_HoverLinkedRotors2():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=-1., target_pos=None):
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
        self.box_size = 4.
        self.half_box_size = (self.box_size/2.)
        self.ceiling = self.target_pos[2]+self.half_box_size
        self.floor = self.target_pos[2]-self.half_box_size

    def is_out_of_bounds(self):
        return (self.sim.pose[2] < self.floor) or (self.sim.pose[2] > self.ceiling)
    
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        if self.is_out_of_bounds():
            reward = 0
        else:
            reward = 2 - abs(self.sim.pose[2] - self.target_pos[2])/self.half_box_size
        return reward
        
        

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds*4) # update the sim pose and velocities
            reward += self.get_reward()
            
            #normalized z position (centered at the target, ideal is 0), z velocity (ideal is 0), z acceleration (ideal is 0)
            pose_all.append([self.target_pos[2] - self.sim.pose[2], self.sim.v[2], self.sim.linear_accel[2]])
        
        next_state = np.concatenate(pose_all)
        if self.is_out_of_bounds():
            done = True    #stop the simulation if outside box
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = [self.sim.pose[2], self.sim.v[2], self.sim.linear_accel[2]] * self.action_repeat 
        return state