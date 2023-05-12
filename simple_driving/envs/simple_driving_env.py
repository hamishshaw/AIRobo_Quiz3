import gym
import numpy as np
import math

import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
from simple_driving.resources.obstacle import Obstacle 
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([40, 40, 1, 1, 5, 5, 10, 10], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self.reached_obstacle = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.obstacle = None
        self.obstacle_object = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0
        self.prev_dist_to_xy = None
        self.prev_dist_obs
        self.prev_dist_diff
        self.prev_position  = None

    def step(self, action):
        # Feed action to the car and get observation of car's state
        if (self._isDiscrete):
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]
        self.car.apply_action(action)
        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          if self._renders:
            time.sleep(self._timeStep)

          carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
          goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
          car_ob = self.getExtendedObservation()

          if self._termination():
            self.done = True
            break
        self._envStepCounter += 1
        # Compute reward as L2 change in distance to goal

        dist_to_goal = math.sqrt(((carpos[0] - goalpos[0]) ** 2 +
                                  (carpos[1] - goalpos[1]) ** 2))
        dist_diff = self.prev_dist_to_goal - dist_to_goal 
        
        #reward = -dist_to_goal
        self.prev_dist_to_goal = dist_to_goal


        obspos, obsorn = self._p.getBasePositionAndOrientation(self.obstacle_object.obstacle)
        dist_to_obs = math.sqrt(((carpos[0] - obspos[0]) ** 2 +
                                  (carpos[1] - obspos[1]) ** 2)) 
        moved =  math.sqrt(((carpos[0] - self.prev_position[0]) ** 2 +
                                  (carpos[1] - self.prev_position[1]) ** 2)) 
        diff_obs = self.prev_dist_obs - dist_to_obs
        
        # calculating reward
        reward = dist_diff * 7

        if (moved < 0.05) & (moved > -0.05) : 
            reward = -5

        #if (self.prev_dist_diff - dist_diff) < 0:
        #    reward -= reward*0.8
        ##if(dist_to_obs < 2.8):
        #    reward = -diff_obs * 10
        self.prev_dist_diff = dist_diff
        self.prev_dist_obs = dist_to_obs   
        self.prev_position = carpos 
        if dist_to_obs < 1 and not self.reached_goal:
            self.done = True
            self.reached_obstacle = True
            reward = -50
        # Done by reaching goal
        if dist_to_goal < 1.5 and not self.reached_goal:
            #print("reached goal")
            self.done = True
            self.reached_goal = True         
            reward = 50

        #done by falling off track
        if carpos[2] < 0:
            self.done = True
            reward = -50
        ob = car_ob
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0
        carpos = self.car.get_observation()

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        self.goal = (x, y)
        self.done = False
        self.reached_goal = False
        
        # Visual element of the goal
        self.goal_object = Goal(self._p, self.goal)
    	
        # select random location for obstacle
        # u changed the obstacle to spawn out of the way as I couldnt get it to reach the goal consistently without hitting the obejct
        xO = -(self.goal[0] - carpos[0])/2 + np.random.choice([1.5,1,0.5,0,-.5,-1,-1.5])
        yO = -(self.goal[1] - carpos[1])/2 + np.random.choice([1.5,1,0.5,0,-.5,-1,-1.5])
        self.obstacle = [xO,yO]
        self.reached_obstacle = False
        
        #reload the obstacle
        self.obstacle_object = Obstacle(self._p,self.obstacle)

        # Get observation to return
        
        pos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        self.prev_dist_to_xy = [ abs(carpos[0]-self.goal[0]), abs(carpos[1]-self.goal[1])]

        self.prev_dist_to_goal = math.sqrt(((carpos[0] - self.goal[0]) ** 2 +
                                           (carpos[1] - self.goal[1]) ** 2))
        self.prev_position = carpos
        # calculating heading diff to goal
        x_diff = xO - carpos[0]
        y_diff = yO - carpos[1]

        heading = math.atan2(y_diff,x_diff)
        #getting current orientation must convert from quarternion
        
        rot = Rotation.from_quat(carorn)
        carheading = rot.as_euler('xyz', degrees=False)
        self.prev_dist_diff = 0
        
        self.prev_head_diff = heading-carheading[2]
        self.prev_dist_obs = math.sqrt(((carpos[0] - xO) ** 2 +
                                           (carpos[1] - yO) ** 2))
        
        car_ob = self.getExtendedObservation()
        #car_ob.append(heading - carheading[2])
        return np.array(car_ob, dtype=np.float32)

    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
            # self.rendered_img.set_data(frame)
            # plt.draw()
            # plt.pause(.00001)

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        # self._observation = []  #self._racecar.getObservation()
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        obstaclepos, obstacleorn = self._p.getBasePositionAndOrientation(self.obstacle_object.obstacle)

        invObsPos, invObsOrn = self._p.invertTransform( obstaclepos, obstacleorn)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)
        ObsPosInCar, goalOrnInCar = self._p.multiplyTransforms(invObsPos, invObsOrn, obstaclepos, obstacleorn)

        observation = [goalPosInCar[0], goalPosInCar[1],ObsPosInCar[0],ObsPosInCar[1]]
        return observation 

    def _termination(self):
        return self._envStepCounter > 2000

    def close(self):
        self._p.disconnect()

