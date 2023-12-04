from .planner import load_model
import torchvision.transforms.functional as TF 
import numpy as np
import torch

class Team:
    agent_type = 'image'

    def __init__(self):

        self.team = None
        self.num_players = None
        self.frame = 1
        self.forward_next = False

        self.planner = True     
    
        self.MSEloss = torch.nn.MSELoss()  
        self.total_loss_puck = 0           
        self.total_loss_No_puck = 0        
        self.total_loss_puck_count = 0     
        self.total_loss_No_puck_count = 0  
      

        self.team = None
        self.num_players = None
        self.current_team = 'not_sure'
        self.rescue_count = [0,0]
        self.rescue_steer = [1,1]
        self.recovery = [False,False]
        self.prev_loc = [[0,0],[0,0]]

        if self.planner:
          self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
          self.Planner = load_model()
          self.Planner.eval()
          self.Planner = self.Planner.to(self.device)

          
    def new_match(self, team: int, num_players: int) -> list:
        self.team, self.num_players = team, num_players
        return ['tux', 'tux']

    
    def to_numpy(self, location):
        return np.float32([location[0], location[2]])

    def _to_image300_400(self, coords, proj, view):
        W, H = 400, 300
        p = proj @ view @ np.array(list(coords) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])

    def _to_image(self, x, proj, view, normalization=True):  


        op = np.array(list(x) + [1])
        p = proj @ view @ op
        x = p[0] / p[-1]  
        y = -p[1] / p[-1]
        aimpoint = np.array([x, y])

        if normalization:
          aimpoint = np.clip(aimpoint, -1, 1) 

        return aimpoint

    def x_intersect(self, kart_loc, kart_front):
        slope = (kart_loc[1] - kart_front[1])/(kart_loc[0] - kart_front[0])
        intersect = kart_loc[1] - (slope*kart_loc[0])
        facing_up_grid = kart_front[1] > kart_loc[1]
        if slope == 0:
            x_intersect = kart_loc[1]
        else:
            try:
                if facing_up_grid:
                    x_intersect = (65-intersect)/slope
                else:
                    x_intersect = (-65-intersect)/slope
            except Exception as e:
                  x_intersect = kart_loc[1]
                  
        return (x_intersect, facing_up_grid)


    def front_flag(self, puck_loc, threshold=2.0):
        x=puck_loc[0]
        return (x>(200-threshold)) and (x<(200+threshold))


    def model_controller(self, puck_loc, location,front,velocity,index):

        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}

        pos_me = location
        front_me = front
        kart_velocity = velocity
        velocity_mag = np.sqrt(kart_velocity[0] ** 2 + kart_velocity[2] ** 2)

        x = puck_loc[0] 
        y = puck_loc[1]  

        x = max(0, min(x, 400))
        y = max(0, min(y, 300))


        if self.current_team == 'not_sure':
            if -58 < pos_me[1] < -50:
                self.current_team = 'red'
            else:
                self.current_team = 'blue'

        x_intersect, facing_up_grid = self.x_intersect(pos_me, front_me)
        
        lean_val = 2
        if -10 < pos_me[0] < 10:
            lean_val = 0
        if facing_up_grid and 9 < x_intersect < 40:
            # if red team
            if self.current_team == 'red':
                x += lean_val
            else:
                x -= lean_val
        if facing_up_grid and -40 < x_intersect < -9:
            # if red team
            if self.current_team == 'red':
                x -= lean_val
            else:
                x += lean_val

        # facing inside goal
        if (not facing_up_grid) and 0 < x_intersect < 10:
            # if red team
            if self.current_team == 'red':
                x += lean_val
            else:
                x -= lean_val
        if (not facing_up_grid) and -10 < x_intersect < 0:
            # if red team
            if self.current_team == 'red':
                x -= lean_val
            else:
                x += lean_val

        if velocity_mag > 20:
            action['acceleration'] = 0.5

        if x < 200:
            action['steer'] = -1
        elif x > 200:
            action['steer'] = 1
        else:
            action['steer'] = 0

        if x < 50 or x > 350:
            action['drift'] = True
            action['acceleration'] = 0.2
        else:
            action['drift'] = False

        if x < 100 or x > 300:
            action['acceleration'] = 0.5

        if self.recovery[index] == True:
            action['steer'] = self.rescue_steer[index]
            action['acceleration'] = 0
            action['brake'] = True
            self.rescue_count[index] -= 2

            if self.rescue_count[index] < 1 or ((-57 < pos_me[1] < 57 and -7 < pos_me[0] < 1) and velocity_mag < 5):
                self.rescue_count[index] = 0
                self.recovery[index] = False
        else:
            if self.prev_loc[index][0] == np.int32(pos_me)[0] and self.prev_loc[index][1] == np.int32(pos_me)[1]:
                self.rescue_count[index] += 5
            else:
                if self.recovery[index] == False:
                    self.rescue_count[index] = 0

            if self.rescue_count[index] < 2:
                if x < 200:
                    self.rescue_steer[index] = 1
                else:
                    self.rescue_steer[index] = -1
            if self.rescue_count[index] > 30 or (y > 200):
                # case of puck near bottom left/right
                if velocity_mag > 10:
                    self.rescue_count[index] = 30
                    self.rescue_steer[index] = 0
                else:
                    self.rescue_count[index] = 20
                self.recovery[index] = True

        self.prev_loc[index] = np.int32(pos_me)

        return action


    def get_instance_coords(instance, object_id=8):

        for x in range(300):
            for y in range(400):
                if instance[x][y] == object_id:
                    return True, x, y

        return False, -1, -1


    def act(self, player_state, player_image, soccer_state = None, heatmap1=None, heatmap2=None):  #REMOVE SOCCER STATE!!!!!!!!

        self.my_team = player_state[0]['kart']['player_id']%2

              
        if self.planner:
          
          image1 = TF.to_tensor(player_image[0])[None]
          image2 = TF.to_tensor(player_image[1])[None]
          
          if self.frame >= 30:  #call the planner
            
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            
            aim_point_image_Player1 = self.Planner(image1)
            aim_point_image_Player2 = self.Planner(image2)
            aim_point_image_Player1 = aim_point_image_Player1.squeeze(0)
            aim_point_image_Player2 = aim_point_image_Player2.squeeze(0)
  
          if self.frame < 30:  
            proj1 = np.array(player_state[0]['camera']['projection']).T
            view1 = np.array(player_state[0]['camera']['view']).T
            proj2 = np.array(player_state[1]['camera']['projection']).T
            view2 = np.array(player_state[1]['camera']['view']).T
            x = np.float32([0,0,0]) 
            aim_point_image_Player1 = self._to_image300_400(x, proj1, view1) 
            aim_point_image_Player2 = self._to_image300_400(x, proj2, view2)
            
        self.frame += 1

      
        Kart_A_front = player_state[0]['kart']['front']
        Kart_A_location = player_state[0]['kart']['location']
        Kart_A_vel = player_state[0]['kart']['velocity']
        pos_A = self.to_numpy(Kart_A_location)
        front_A =self.to_numpy(Kart_A_front)
        
        Kart_B_front = player_state[1]['kart']['front']
        Kart_B_location = player_state[1]['kart']['location']
        Kart_B_vel = player_state[0]['kart']['velocity']
        pos_B = self.to_numpy(Kart_B_location)
        front_B = self.to_numpy(Kart_B_front)

       
        action_A = self.model_controller(aim_point_image_Player1,pos_A,front_A,Kart_A_vel,0)
        action_B = self.model_controller(aim_point_image_Player2,pos_B,front_B,Kart_B_vel,1)


        ret = [action_A,action_B]

        return ret
