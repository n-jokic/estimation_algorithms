# -*- coding: utf-8 -*-

import numpy as np


class particle():
    def __init__(self, x_state):
        self.x_current = x_state
        self.x_new = np.zeros(x_state.shape)

    def prediction(self):
        dt = 0.01
        x_state = self.x_current
        sigma = 0.01
        self.x_new = np.array([x_state[0] + (np.cos(x_state[6]) * x_state[1] - np.sin(x_state[6]) * x_state[3]) * dt,
                               x_state[1] + x_state[2] * dt,
                               x_state[2] + np.random.randn()*sigma,
                               x_state[3] + (np.cos(x_state[6]) * x_state[3] + np.sin(x_state[6]) * x_state[1]) * dt,
                               x_state[4] + x_state[5] * dt,
                               x_state[5] + np.random.randn()*sigma,
                               x_state[6] + np.random.randn()*sigma
                               ])

        self.x_current = self.x_new

    def likelihood(self, z):
        sigma = 0.05
        x_obs = z[0]
        acc_x_obs = z[1]
        y_obs = z[2]
        acc_y_obs = z[3]
        theta_obs = z[4]

        x_pred = self.x_current[0]
        acc_x_pred = self.x_current[1]
        y_pred = self.x_current[3]
        acc_y_pred = self.x_current[4]
        theta_pred = self.x_current[5]

        sigma_pos = 1

        
        return np.exp((x_pred-x_obs)**2/2/sigma_pos)*np.exp((y_pred-y_obs)**2/2/sigma_pos)*\
            np.exp((acc_y_pred-acc_y_obs)**2/2/sigma)*np.exp((acc_x_pred-acc_x_obs)**2/2/sigma)*\
            np.exp((theta_pred-theta_obs)**2/2/sigma)


class particle_filter():
    
    def __init__(self, **kwargs):
        
        if('sensor_data' in kwargs):
            self.sensor_data = kwargs['sensor_data']
        else:
            self.sensor_data = np.array([])
        
        if('N_particles' in kwargs):
            self.Np = kwargs['N_particles']
        else:
            self.Np = 10

        self.particles = [particle(x_state=np.zeros((7, ))) for i in range(self.Np)]
            
        if('algorithm' in kwargs):
            self.algorithm = kwargs['algorithm']
        else:
            self.algorithm = "no_resampling"
            
        self.weights = np.ones((self.Np, 1))/self.Np
        
        self.t = 1
            
        self.old_particles = self.particles
        self.old_weights = self.weights

    def setSensorData(self, data):
        self.sensor_data = np.array(data)
    
    def setNumberParticles(self, Np):
        self.Np = Np
        self.particles = [particle() for i in range(self.Np)]
        self.weights = np.ones((self.Np, 1))/Np
        
    def setCurrentStep(self, t):
        self.t = t
    def setAlgorithm(self, alg):
        self.algorithm = alg
        
    def resetParticles(self, **kwargs):
        self.particles = [particle(x_state=np.zeros((7, ))) for i in range(self.Np)]
            
        self.old_particles = self.particles
        
        self.weights = np.ones((self.Np, 1))/self.Np
        self.old_weights = self.weights
        self.t = 1
        
    def particle_filtering(self):
        
        if self.t > len(self.sensor_data):
            print('No more sensor data')
            return None
        
        
        z = self.sensor_data[self.t]
        self.t += 1
        
        for i in range(self.Np):
            self.particles[i].prediction()
            
            self.weights[i] *= self.particles[i].likelihood(z)
            
        self.weights = self.weights/np.sum(self.weights)
            
        estimation = np.zeros((7, ))
        
        for idx, part in enumerate(self.particles):
            estimation += part.getEstimation()*self.weights[idx]
        
        self.old_particles = self.particles 
        self.old_weights = self.weights
        
        self.particles, self.weights = self.__update_particles(self.algorithm)
        
        return estimation
        
    def __update_particles(self, algorithm):
        if algorithm == "no_resampling":
            return self.__no_resampling()
        if algorithm == "iterative_resampling":
            return self.__iterative_resampling()
        if algorithm == "dynamic_resampling":
            return self.__dynamic_resampling()
        
        print('Wrong algorithm specified')
        return self.particles, self.weights
    
    def __no_resampling(self):

        return self.particles, self.weights
        
    def __iterative_resampling(self):
        new_particles = [0]*self.Np
        new_weights =  np.ones((self.Np, 1))/self.Np

        indices = np.random.choice(self.Np, self.Np, p=self.weights.flatten())
       
        for idx in range(self.Np):
            idy = indices[idx]
            x_state = self.particles[idy].getValues()
            new_particles[idx] = particle(x_state=x_state)
           
        return new_particles, new_weights
    
    def __dynamic_resampling(self):
        
        effective_size =1/np.sum(self.weights**2)

        if(effective_size < self.Np/2):
            return self.__iterative_resampling()
        else: 
            return self.__no_resampling()