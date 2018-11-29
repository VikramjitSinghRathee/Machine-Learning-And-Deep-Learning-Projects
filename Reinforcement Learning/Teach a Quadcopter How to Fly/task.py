import numpy as np
from physics_sim import PhysicsSim
from math import e, sqrt, pi

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
        ## PREVIOUS
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum() # getting only x,y,z from pose. summing values for each direction.
        
        
        ## (UPDATED CODE!!)
        if self.sim.pose[2] >= self.target_pos[2]:
            bonus = 10.
        else:
            bonus = 0.
        
        if self.sim.v[2] > 0:
            z_vel_bonus = 10*self.sim.v[2]
        else:
            z_vel_bonus = -10
            
        reward = -(abs(self.sim.pose[2]-self.target_pos[2]))+ bonus + z_vel_bonus
        
        
        '''
        reward = -(abs(self.sim.pose[2]-self.target_pos[2]))
        if self.sim.v[2]>0:
            reward = 10*self.sim.v[2]
        else:
            reward = -10
            
        if(self.sim.pose[2] > self.target_pos[2]):
            reward += 10
        '''
            
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        ## each set of rotor speed/action is repeated 3 times, states and rewards updated accordingly.
        ## keeps track if the episode is done or not
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
                                                        # given a rotor speed performs simulation and updates pose and velocities
                                                        # next_timestep also gives info when episode is done.
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done     # next_state is 18 element array

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        #print([self.sim.pose] * self.action_repeat) ## added
        return state