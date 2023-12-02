import numpy as np
import torch
from numpy.linalg import norm
from . import planner
import torch.nn.functional as F
from os import path
from .import utils

class Team:
    TARGET_SPEED = 15
    DRIFT_ANGLE = 20
    BRAKE_ANGLE = 30
    DEFENSE_RADIUS = 40

    PUCK = None
    PUCK_T = 0

    def __init__(self, player_id=0, kart='wilber'):
        self.player_id = player_id
        self.kart = kart
        self.team = player_id % 2
        self.own_goal = np.float32([0, -65 if self.team == 0 else 65])
        self.goal = np.float32([0, 64 if self.team == 0 else -64])
        self.model = planner.Planner()
        model_path = path.join(path.dirname(path.abspath(__file__)), 'planner.th')
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        self.offense = (player_id == 0)
        self.pucklock = True
        self.puck = np.float32([0, 0])
        self.t = 0

        self.last_puck = None
        self.last_loc = None 

    def detect_puck(self, image):
        with torch.no_grad():
            img = F.interpolate(torch.from_numpy(image / 255.0).float().permute(2, 0, 1)[None], size=(75, 100))
            img = img[0, :, 27:, :]
            dets, depth, is_puck = self.model.detect(img)
            dets = [(de[0], (de[1] + 0) * 4, (de[2] + 27) * 4, de[3], de[4]) for de in dets[0]]
        return dets, depth, is_puck

    def update_puck(self, dets, is_puck, player_info):
        if is_puck > 0 and dets and dets[0][0] > 2:
            proj = np.array(player_info.camera.projection).T @ np.array(player_info.camera.view).T
            coords = dets[0]
            return utils.center_to_world(coords[1], coords[2], 400, 300, proj)[[0, 2]]
        return None

    def compute_aim(self, puck, kart, front):
        u = front - kart
        u /= norm(u)
        if self.offense:
            return self.compute_offense_aim(puck, kart, u)
        else:
            return self.compute_defense_aim(puck, kart, u)

    def compute_offense_aim(self, puck, kart, u):
        puck_goal = self.goal - puck
        puck_goal /= norm(puck_goal)
        kart_puck_dist = norm(puck - kart)
        return puck - puck_goal * kart_puck_dist / 2, u

    def compute_defense_aim(self, puck, kart, u):
        own_goal_puck = puck - self.own_goal
        own_goal_puck_dist = norm(own_goal_puck)
        if own_goal_puck_dist < self.DEFENSE_RADIUS:
            aim = puck - 1 * own_goal_puck / own_goal_puck_dist
        elif np.abs(kart[1]) < np.abs(self.own_goal[1]):
            aim = self.own_goal
            u = -u
        else:
            aim = self.goal
        return aim, u

    def get_action(self, aim, kart, u, vel, theta):
        v = aim - kart
        v /= norm(v)
        signed_theta = -np.sign(np.cross(u, v)) * np.sign(np.dot(u, vel)) * np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
        steer = signed_theta / 8
        brake = self.BRAKE_ANGLE <= abs(signed_theta)
        drift = self.DRIFT_ANGLE <= abs(signed_theta) and not brake
        accel = 1 if norm(vel) < self.TARGET_SPEED and not brake else 0
        return {'steer': steer, 'acceleration': accel, 'brake': brake, 'drift': drift, 'nitro': False, 'rescue': False}

    def act(self, image, player_info):
        dets, depth, is_puck = self.detect_puck(image)
        front = np.float32(player_info.kart.front)[[0, 2]]
        kart = np.float32(player_info.kart.location)[[0, 2]]
        vel = np.float32(player_info.kart.velocity)[[0, 2]]
        puck = self.update_puck(dets, is_puck, player_info)

        if puck is None and self.t < 60:
            puck = np.float32([0, 0])
        elif puck is not None:
            Team.PUCK = puck
            Team.PUCK_T = self.t

        aim, u = self.compute_aim(puck, kart, front)
        return self.get_action(aim, kart, u, vel, norm(front - kart))
