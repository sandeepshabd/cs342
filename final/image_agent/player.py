import numpy as np
import torch
from planner import Planner, load_model
from torchvision.transforms.functional import to_tensor

class Team:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
        self.planner = Planner().to(self.device).eval()
        self.frame = 1
        self.current_team = 'not_sure'
        self.rescue_count = [0, 0]
        self.rescue_steer = [1, 1]
        self.recovery = [False, False]
        self.prev_loc = [[0, 0], [0, 0]]

    def new_match(self, team: int, num_players: int) -> list:
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players
    
    @staticmethod
    def to_numpy(location):
        return np.float32([location[0], location[2]])

    @staticmethod
    def _to_image300_400(coords, proj, view):
        W, H = 400, 300
        p = proj @ view @ np.array(list(coords) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])
    

    
    def act(self, player_state, player_image):
        # Setup projection and view matrices
        proj = np.array(player_state[0]['camera']['projection']).T
        view = np.array(player_state[0]['camera']['view']).T
        
        actions = []
        for i, player in enumerate(player_state):
            image_tensor = to_tensor(player_image[i])[None].to(self.device)
            aim_point_image = self.planner(image_tensor).squeeze(0)
            
            pos = self.to_numpy(player['kart']['location'])
            front = self.to_numpy(player['kart']['front'])
            vel = self.to_numpy(player['kart']['velocity'])
            action = self.model_controller(aim_point_image, pos, front, vel, i)
            actions.append(action)
        
        self.frame += 1
        return actions

    # The model_controller function would be defined here.
    # It's assumed to be a complex function that calculates the next action based on inputs.
