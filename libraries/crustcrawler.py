

import gym
import random
import numpy as np

from gym import spaces
from spatialmath import base
from roboticstoolbox.backends.PyPlot import PyPlot

import spatialmath as sm
import spatialgeometry as sg
import roboticstoolbox as rtb

from libraries import utils


MIN_DISTANCE_TARGET = 0.01
MIN_ANGULAR_TARGET = 0.03490658503989

MIN_ACHIEVED_GOAL = 1

MIN_EFFECTOR_HEIGHT = 0.0255

pose_alvo = utils.ler_arquivo_pose("./input/pose.txt")


# configuracoes das juntas
link_configs = {
	'base'		: (  2.0, 280.0),
	'shoulder'	: ( 35.0, 180.0),
	'forearm' 	: ( 35.0, 250.0),
	'wrist' 	: (  0.0, 300.0),
	'grip'		: (150.0, 220.0),
}


class CrustCrawlerEnv(gym.Env):

    def __init__(self):
        super(CrustCrawlerEnv, self).__init__()
        self.env = PyPlot()
        self.env.launch('CrustCrawler')
        
        # Definição dos links usando os parâmetros DH
        Ls = []
        Ls.append(rtb.RevoluteDH(a=0.055, 
                                 alpha=-np.pi/2, 
                                 d=0.092, 
                                 offset=0,        
                                 qlim=np.deg2rad(list(link_configs['base']))))
       
        Ls.append(rtb.RevoluteDH(a=0.173, 
                                 alpha=np.pi, 	 
                                 d=0,     
                                 offset=-np.pi,   
                                 qlim=np.deg2rad(list(link_configs['shoulder']))))
        
        Ls.append(rtb.RevoluteDH(a=0.020, 
                                 alpha=np.pi/2,  
                                 d=0,     
                                 offset=-np.pi/2+np.pi/6, # deslocamento de 30º
                                 qlim=np.deg2rad(list(link_configs['forearm']))))
        
        Ls.append(rtb.RevoluteDH(a=0.0, 
                                 alpha=0.0, 	 
                                 d=0.068, 
                                 offset=0,        
                                 qlim=np.deg2rad(list(link_configs['wrist']))))

        # Definir a base do robô (posição e orientação)
        base = sm.SE3(0, 0, 0.02) * sm.SE3.Rz(0)  # Base deslocada para (1, 2, 3) e rotacionada em pi/4 ao redor de Z

        # Criação do robô
        self.robot = rtb.DHRobot(Ls, name="CrustCrawler", base=base)
        self.env.add(self.robot)

        self.target_object_1, self.target_position_1, \
            self.target_orientation_1, self.target_axes_1 = \
                self._creates_target_object_in_scene_1()
        
        ax = self.env.ax
        self.draw_axes(ax, self.target_object_1)


        self.obs_low = np.concatenate((self.robot.qlim[0], 
                                       np.array([-1.0, -1.0, -1.0]), 
                                       np.array([-1.0, -1.0, -1.0]), 
                                       np.array([-1.0, -1.0, -1.0]),
                                       np.array([-np.pi, -np.pi, -np.pi]),
                                       np.array([-np.pi, -np.pi, -np.pi]),
                                       np.array([-np.pi, -np.pi, -np.pi]),
                                       np.array([0.0]),
                                       np.array([0.0]),
                                       np.array([0.0])))
        
        self.obs_high = np.concatenate((self.robot.qlim[1], 
                                        np.array([+1.0, +1.0, +1.0]),
                                        np.array([+1.0, +1.0, +1.0]),
                                        np.array([+1.0, +1.0, +1.0]),
                                        np.array([+np.pi, +np.pi, +np.pi]),
                                        np.array([+np.pi, +np.pi, +np.pi]),
                                        np.array([+np.pi, +np.pi, +np.pi]),
                                        np.array([2.0]),
                                        np.array([3.0 * +np.pi]),
                                        np.array([1.0])))
        
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, shape=(25,), dtype=np.float64)
        
        self.action_space = spaces.Discrete(3)

        self.dt = 0.01        
        self.distances_list = []
        self.orientations_list = []

        self.episode = 0
        self.previous_fitness = None


    def reset(self):
        self.episode += 1
        self.current_step = 0
        self.count_achieved_goal = 0

        self.collided = False
        self.target_achieved = False
        
        self.distances_list.clear()
        self.orientations_list.clear()
        self._reset_robot_position()

        self.target_object = self.target_object_1
        self.target_position = self.target_position_1
        self.target_orientation = self.target_orientation_1

        obs = self._get_obs()
        return obs, ""

    def step(self, actions):

        q_new = self.getP(actions)
        self.robot.q[0:4] += q_new

        self.robot.q = np.clip(self.robot.q[:4], self.obs_low[:4], self.obs_high[:4]) # Faz a limitação das velocidades de acordo com a especificação para evitar trepidação
        self.env.step(self.dt)

        self.current_step += 1
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, {}


    def getP(self, actions):
        """ Obtem os novos angulos das juntas
        """             

        new_angles = []
        
        for j, action in enumerate(actions):

            if j == 0:
                delta_theta = 0.034906
            if j == 1:
                delta_theta = 0.0250
            if j == 2:
                delta_theta = 0.0260
            if j == 3:
                delta_theta = 0.0270

            
            if action.item() == 0:
                new_angles.append(-delta_theta)
            elif action.item() == 1:
                new_angles.append(0)
            elif action.item() == 2:
                new_angles.append(+delta_theta)

        return np.array(new_angles)


    def _compute_reward(self):
        """ A função de recompensa adaptada do artigo, levando em consideração a 
            fitness de posição e orientação. 
        """
        done = False
        collision = False
        collision_reward = 0.0
        completion_reward = 0.0

        pose_error, orientation_error = self._get_final_pose_by_forward_kinematics()
        collision = self._check_collision()
        
        # Fitness atual (com base no erro de posição e orientação)
        position_fitness = np.linalg.norm(pose_error)
        orientation_fitness = np.linalg.norm(orientation_error[0])

        self.distances_list.append(position_fitness)
        self.orientations_list.append(orientation_fitness)

        effector_position = self._get_current_effector_position()
        if effector_position[-1] < MIN_EFFECTOR_HEIGHT:
            collision =  True

        # Calcula os pesos para o controle de posição
        position_weight = 0.90
        orientation_weight = 0.10

        # Calcula a fitness total ponderando os erros de posição e orientação
        total_fitness = (position_fitness * position_weight) + (orientation_fitness * orientation_weight)

        # Caso o alvo seja atingido (dentro de um limite de erro aceitável)
        if position_fitness <= MIN_DISTANCE_TARGET and orientation_fitness <= MIN_ANGULAR_TARGET and collision is False:
            self.count_achieved_goal += 1
            self.target_achieved = True
        else:
            self.count_achieved_goal = 0  # reset se houve movimento
            self.target_achieved = False

        if self.count_achieved_goal >= MIN_ACHIEVED_GOAL:
            done = True

        # Verifica se é a primeira iteração (não tem fitness anterior)
        if self.previous_fitness is None:
            self.previous_fitness = total_fitness
        
        # Função de recompensa baseada na diferença de fitness 
        fitness_difference = self.previous_fitness - total_fitness 
        reward = 100 * (2 / (1 + np.exp(-10 * (fitness_difference / MIN_DISTANCE_TARGET))) - 1)

        # Se colidiu, penalidade
        if collision:
            self.collided = True 
        else:
            self.collided = False

        reward += (completion_reward + collision_reward)

        # Atualiza a fitness anterior para a próxima iteração
        self.previous_fitness = total_fitness

        return reward, done

    def _get_current_effector_position(self):
        effector_pose = self._get_current_effector_object()
        return effector_pose.t

    def _get_current_effector_object(self):
        return self.robot.fkine(self.robot.q)

    def _get_final_pose_by_forward_kinematics(self):
        wTe = self.robot.fkine(self.robot.q)
        wTep = self.target_object
        wTe = wTe.A
        wTep = wTep.A
        eTep = np.linalg.inv(wTe) @ wTep
        
        pose = np.empty(3)
        # Translational error
        pose = eTep[:3, -1]

        orie = np.empty(3)
        # Angular error
        orie = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
        
        return pose, orie

    def _calculate_target_distance_difference(self):
        effector_pose = self._get_current_effector_object()
        return np.linalg.norm(effector_pose.t - self.target_position)


    def _creates_target_object_in_scene_1(self):
        Tx = sm.SE3.Tx(pose_alvo['px'])
        Ty = sm.SE3.Ty(pose_alvo['py']) 
        Tz = sm.SE3.Tz(pose_alvo['pz'])
        Rx = sm.SE3.Rx(pose_alvo['rx']) 
        Ry = sm.SE3.Ry(pose_alvo['ry']) 
        Rz = sm.SE3.Rz(pose_alvo['rz'])
        
        target_object = Tx * Ty * Tz * Rx * Ry * Rz
        target_position = target_object.t

        target_orientation = sm.base.tr2rpy(target_object.R, unit="rad")

        target_axes = sg.Axes(length=0.05, pose=target_object)   
        return target_object, target_position, target_orientation, target_axes
    
    def _get_current_effector_object(self):
        return self.robot.fkine(self.robot.q)

    def _get_current_effector_orientation(self):
        effector_pose = self._get_current_effector_object()
        rotation_matrix = effector_pose.R
        euler_angles = sm.base.tr2rpy(rotation_matrix, unit="rad")
        return euler_angles

    def _get_obs(self):
        effector_pose = self._get_current_effector_object()
        effector_orientation = self._get_current_effector_orientation()
        diff_pose, diff_orie = self._get_final_pose_by_forward_kinematics()       

        distance_difference = np.linalg.norm(diff_pose)
        orientation_difference = np.linalg.norm(diff_orie)
        collision = 1 if self.collided else 0
        
        return np.concatenate((self.robot.q[:4], 
                               effector_pose.t, 
                               self.target_position, 
                               diff_pose,
                               effector_orientation,
                               self.target_orientation,
                               diff_orie,
                               [distance_difference],
                               [orientation_difference], 
                               [collision]))

    def _reset_robot_position(self):
        self.robot.q = [2.4609141, 1.8762289, 2.4870942, 2.6179938]	
        self.robot.q += 0.05 * np.random.uniform(-1, 1, 4)
    
    # Função para desenhar uma esfera
    def draw_sphere(self, ax, position, radius=0.01, color='red'):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + position[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + position[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + position[2]

        ax.plot_surface(x, y, z, color=color, alpha=0.7)
