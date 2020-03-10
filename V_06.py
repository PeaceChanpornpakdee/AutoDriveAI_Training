# Make sure to have the server side running in V-REP: 
# in a child script of a V-REP scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19998)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

import vrep
import sys
import numpy as np
import time
import random
import math

'''----------------------------------------------------------------------------------------------------------------------------'''

class VREP_env:
    
    def __init__(self):

        self.clientID = 99
        self.motor_name = ['FL','FR','RL','RR']

        self.robot_handle = 0
        self.goal_handle = 0
        self.motor_handle = []
        self.cube_handle = []
        self.sensor_handle = []

        self.position_robot = []
        self.position_goal = []
        self.velocity_robot = []
        self.angvelocity_robot = []
        self.angle_robot = []

        self.l = []
        self.action_list = [0,0,0,0,0,0,0,0,0]

        self.done = False
        self.state = []

    '''----------------------------------------------------------------------------------------------------------------------------'''

    def randomPos(self):

        far = 1.50
        obj = 12
        workspace = 3.5

        repeat = False
        overtime = False

        while (True):
            self.l = []
            ti = time.time()
            for i in range(0, obj):
                while (True):
                    repeat = False
                    overtime = False
                    if (time.time() - ti > 0.1):
                        overtime = True
                        break

                    x = random.uniform(-workspace, workspace)
                    y = random.uniform(-workspace, workspace)
                    for i in range(0, len(self.l)):
                        if (abs(x - self.l[i][1]) < far) and (abs(y - self.l[i][2]) < far):
                            repeat = True
                            break

                    if (abs(x - 0.0) < far) and (abs(y - 0.0) < far):
                        repeat = True

                    if (repeat == False):
                        break

                self.l.append([x + y, x, y])
                if (overtime == True):
                    break
            if (repeat == False and overtime == False):
                break


        
#--------------------------------------------------------------------------------------------------------------------------------        

    def setUp(self):
     
        '''////////// CONNECT //////////'''
        vrep.simxFinish(-1) # just in case, close all opened connections    
        self.clientID=vrep.simxStart('127.0.0.1',19998,True,True,5000,5) # Connect to V-REP   
        
        if self.clientID!=-1:
            pass
            #print ('Connected to remote API server')
        else:
            print ('Connection Failed')
            sys.exit('Could not connect')
            
        self.done = False
        
        '''////////// HANDLE //////////'''    
        _, self.goal  = vrep.simxGetObjectHandle(self.clientID,'Goal',vrep.simx_opmode_oneshot_wait)
        _, self.robot = vrep.simxGetObjectHandle(self.clientID,'Omnirob',vrep.simx_opmode_oneshot_wait)
        for i in range(0,4):
            self.motor_handle.append(vrep.simxGetObjectHandle(self.clientID,'Omnirob_'+self.motor_name[i]+'wheel_motor',vrep.simx_opmode_oneshot_wait)[1])

        for i in range(0, 9 + 1):
            self.cube_handle.append(vrep.simxGetObjectHandle(self.clientID, 'Cuboid' + str(i),vrep.simx_opmode_oneshot_wait)[1])


        for i in range(0,18):
            self.sensor_handle.append( vrep.simxGetObjectHandle(self.clientID,'sensor'+str(i+1),vrep.simx_opmode_oneshot_wait)[1] )

        '''////////// SET POSITION & ANGLE //////////'''
        vrep.simxSetObjectOrientation(self.clientID,self.robot,-1,[0,0,math.pi],vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(self.clientID,self.goal ,-1,[0,0,random.uniform(0,2*math.pi)],vrep.simx_opmode_oneshot)

        self.l = []
        self.randomPos()

        z = random.randint(0,10)
        z = -10.25 if z in [1,5,8,9] else 0.25
        vrep.simxSetObjectPosition(self.clientID,self.robot,-1,[ self.l[10][1],self.l[10][2],0.28 ],vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(self.clientID,self.goal, -1,[  self.l[11][1],self.l[11][2],0.00 ],vrep.simx_opmode_oneshot)

        for i in range(0, 9 + 1):
            vrep.simxSetObjectPosition(self.clientID, self.cube_handle[i], -1,
                                                   [self.l[i][1], self.l[i][2],z], vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.cube_handle[i], -1,
                                                      [0, 0, random.uniform(0, 2 * math.pi)], vrep.simx_opmode_oneshot)

        _, self.velocity_robot, self.angvelocity_robot = vrep.simxGetObjectVelocity(self.clientID,self.robot, -1)
        _, self.position_robot = vrep.simxGetObjectPosition   (self.clientID,self.robot,-1,vrep.simx_opmode_streaming)
        _, self.position_goal  = vrep.simxGetObjectPosition   (self.clientID,self.goal ,-1,vrep.simx_opmode_streaming)
        _, self.angle_robot    = vrep.simxGetObjectOrientation(self.clientID,self.robot,-1,vrep.simx_opmode_streaming)

        '''////////// START //////////'''
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_continuous)
        self.pre_distance2goal = 0.0

# --------------------------------------------------------------------------------------------------------------------------------

    def getSensor(self):
        sensor_read = []
        for i in range(0, 18):
            _, detectionState_n, detectedPoint_n, detectedObjectHandle_n, detectedSurfaceNormalVector_n = vrep.simxReadProximitySensor(
                self.clientID, self.sensor_handle[i], vrep.simx_opmode_streaming)
            sensor_read.append(np.clip(np.linalg.norm(detectedPoint_n), 0.0, 2.0))
        return sensor_read

    #--------------------------------------------------------------------------------------------------------------------------------

    def observe(self):

        _, self.position_goal  = vrep.simxGetObjectPosition(self.clientID,self.goal ,-1,vrep.simx_opmode_buffer)
        _, self.position_robot = vrep.simxGetObjectPosition(self.clientID,self.robot,-1,vrep.simx_opmode_buffer)
        _, self.velocity_robot, self.angvelocity_robot = vrep.simxGetObjectVelocity(self.clientID, self.robot, -1)
        _, self.angle_robot    = vrep.simxGetObjectOrientation(self.clientID,self.robot,-1,vrep.simx_opmode_buffer)
        self.state = []
        dx = self.position_robot[0]-self.position_goal[0]
        dy = self.position_robot[1]-self.position_goal[1]
        s1 = np.array([dx,dy])
        s2 = np.array(self.getSensor())
        self.state = np.concatenate([s1,s2])



        return np.array( self.state )
    
#--------------------------------------------------------------------------------------------------------------------------------        
        
    def action(self,v):
        a = [0,0,0,0]
        a[0] = v[0]-v[1]
        a[1] = -v[0]-v[1]
        a[2] = v[0] + v[1]
        a[3] = -v[0] + v[1]
        for i in range(0,4):
            vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handle[i] , 5*a[i], vrep.simx_opmode_streaming)

#--------------------------------------------------------------------------------------------------------------------------------        

#--------------------------------------------------------------------------------------------------------------------------------        

    def reward(self):

        dx = self.position_robot[0] - self.position_goal[0]
        dy = self.position_robot[1] - self.position_goal[1]

        distance2goal = np.sqrt((dx) ** 2 + (dy) ** 2)

        r = 5.0

        if distance2goal <= 0.5:
            r += 10

        if ( self.pre_distance2goal - distance2goal) >= 0.01:
            r += 5.0 * (self.pre_distance2goal - distance2goal)
        elif (self.pre_distance2goal - distance2goal) < 0.01:
            r += -5.0
        self.pre_distance2goal = distance2goal

        return r

#--------------------------------------------------------------------------------------------------------------------------------        

    def step(self,v):
        
        self.action(v)

        obs = self.observe()
        rew = self.reward()

        end = self.done
        vrep.simxSetObjectOrientation(self.clientID, self.robot, -1, [0, 0, math.pi], vrep.simx_opmode_oneshot)
        return obs, rew, end


#--------------------------------------------------------------------------------------------------------------------------------        

    def end(self):
        
        self.action([0,0,0])
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        vrep.simxFinish(-1)
    

#--------------------------------------------------------------------------------------------------------------------------------        


    def main(self):

    
        self.setUp()
        while(1):
            print(self.getSensor())
        self.end()



#--------------------------------------------------------------------------------------------------------------------------------


#env = VREP_env()
#env.main()