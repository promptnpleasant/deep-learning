import numpy as np
from physics_sim import PhysicsSim

class RotorsLinkedTask():
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

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speed):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        rotor_speeds = rotor_speed*4
        #pose_all = []
        state_data_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            #pose_all.append(self.sim.pose)
            state_data_all.append(np.concatenate([self.sim.pose] + [self.sim.v]))
            #print('state_data_all',state_data_all)
        next_state = np.concatenate(state_data_all)
        #print(next_state)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #print('pose', self.sim.pose)        
        #print('v',self.sim.v)
        #print('action_repeat',self.action_repeat)
        state = np.concatenate([np.concatenate([self.sim.pose] + [self.sim.v])] * self.action_repeat)        
        #print('state',state)
        return state
    
class MyTask():
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

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.ideal_hover_rotor_speeds = np.array([404.,404.,404.,404.])
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self,rotor_speeds):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = 2
        reward -= .1*abs(self.sim.pose[0]-self.target_pos[0])
        reward -= .1*abs(self.sim.pose[1]-self.target_pos[1])
        reward -= .5*abs(self.sim.pose[2]-self.target_pos[2])
        reward -= .5*abs(self.sim.pose[5])
        reward -= .01*abs(self.sim.v[0])
        reward -= .01*abs(self.sim.v[1])
        reward -= .05*abs(self.sim.v[2])
        reward -= .1*(abs(rotor_speeds-self.ideal_hover_rotor_speeds)).sum()
        av_rotor=np.mean(rotor_speeds)
        reward -= .5*(abs(rotor_speeds-av_rotor)).sum()
        return np.tanh(reward)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        #pose_all = []
        state_data_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            #pose_all.append(self.sim.pose)
            state_data_all.append(np.concatenate([self.sim.pose] + [self.sim.v]))
            #print('state_data_all',state_data_all)
        next_state = np.concatenate(state_data_all)
        #print(next_state)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #print('pose', self.sim.pose)        
        #print('v',self.sim.v)
        #print('action_repeat',self.action_repeat)
        state = np.concatenate([np.concatenate([self.sim.pose] + [self.sim.v])] * self.action_repeat)        
        #print('state',state)
        return state

class Task():
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

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state