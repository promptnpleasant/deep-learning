import numpy as np
from physics_sim import PhysicsSim

class task_Hover_Full():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=20., target_pos=None):
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

        self.state_size = self.action_repeat * 18
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.sphere_radius = 2.0

    def get_distance_squared(self):
        return ((self.sim.pose[0] - self.target_pos[0])**2 + 
                (self.sim.pose[1] - self.target_pos[1])**2 + 
                (self.sim.pose[2] - self.target_pos[2])**2)
    
    def is_out_of_bounds(self):
        return (self.get_distance_squared() > (self.sphere_radius)**2)
                
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        if self.is_out_of_bounds():
            reward = 0.0
        else:
            reward = (self.get_distance_squared()/(self.sphere_radius**2))   #reward = 1 to 0  as distance goes from target to distance r^2 away from the target
            reward += 0.2 - 0.2 * np.cos(self.sim.pose[5])   #adding reward for keeping the yaw at 0.
            reward += (0.2 - 0.4 * np.exp(-abs(sum(self.sim.v))))   #reward = 1 to 0  as velocities go from 0 to infinity
        return reward
        
                                            
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(np.concatenate([self.sim.pose] + [self.sim.v] + [self.sim.linear_accel] + [self.sim.angular_v] + [self.sim.angular_accels]))
        next_state = np.concatenate(pose_all)
        
        if self.is_out_of_bounds():
            done = True    #stop the simulation if outside box
        elif done:
            print('task conquered!')
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([np.concatenate([self.sim.pose] + [self.sim.v] + [self.sim.linear_accel] +
                                               [self.sim.angular_v] + [self.sim.angular_accels])] * self.action_repeat) 
        return state