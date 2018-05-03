import numpy as np
from physics_sim import PhysicsSim

class task_TakeOffLinkedRotors():
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

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 5. + 0.1*(self.sim.pose[2] - self.target_pos[2]) #reward of 1 when at height, increasingly negative when below, increasingly positive when above
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds*4) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append([self.sim.pose[2], self.sim.v[2], self.sim.linear_accel[2]])
        next_state = np.concatenate(pose_all)
        if self.sim.pose[2] >= self.target_pos[2]:
            done = True    #stop the simulation if you get to the goal
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = [self.sim.pose[2], self.sim.v[2], self.sim.linear_accel[2]] * self.action_repeat 
        return state