import math 
from .planner import Planner, load_model
import torchvision.transforms.functional as TF 
import numpy as np
import torch

def limit_period(angle):
    # turn angle into -1 to 1 
      return angle - torch.floor(angle / 2 + 0.5) * 2 
    
def extract_featuresV3(pstate, soccer_state, opponent_state, team_id):
      # features of ego-vehicle
      kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
      kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
      kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
      kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

      # features of soccer 
      puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
      kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
      kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 

      kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

      # features of score-line 
      goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

      puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

      features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, 
          goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference, 
          puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)

      return features 

class Team:
    agent_type = 'image'


    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.frame = 1
        self.forward_next = False
        
        ##-------------------------------------------
        self.planner = True     #set to true to use the planner, debugging purposes
        self.DEBUG = False      #SET TO TRUE TO DEBUG


        self.MSEloss = torch.nn.MSELoss()  #for DEBUGGING 
        self.total_loss_puck = 0           #for DEBUGGING
        self.total_loss_No_puck = 0        #for DEBUGGING
        self.total_loss_puck_count = 0     #for DEBUGGING
        self.total_loss_No_puck_count = 0  #for DEBUGGING
        
        ##--------------------------------------------
        
        self.rescue_count = [0,0]
        self.rescue_steer = [1,1]
        self.recovery = [False,False]
        self.prev_loc = [[0,0],[0,0]]
        
        if not self.planner:
          print ("****** planner not used  *******")

        if self.planner:

          print ("****** planner *******")
          
          self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
          self.Planner = load_model()
          self.Planner.eval()
          self.Planner = self.Planner.to(self.device)
          print (self.Planner)

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        print('----------------starting new match---------------------')
        self.team, self.num_players = team, num_players
        return ['tux', 'tux']
      
    def to_numpy(self, location):
        return np.float32([location[0], location[2]])


      
    def to_screen_coordinates(self, coords, proj, view, viewport_size=(400, 300)):
        # Unpack the viewport dimensions
        width, height = viewport_size
        # Perform the matrix multiplications
        p = proj @ view @ np.append(coords, 1)
        # Convert to screen coordinates
        screen_x = (p[0] / p[3] + 1) * (width / 2)
        screen_y = (1 - p[1] / p[3]) * (height / 2)
        return np.array([screen_x, screen_y])

    def x_intersect(kart_loc, kart_front, grid_lines_y=(65, -65)):
        try:
            slope = (kart_front[1] - kart_loc[1]) / (kart_front[0] - kart_loc[0])
            intersect_y = kart_loc[1] - (slope * kart_loc[0])
            facing_up_grid = kart_front[1] > kart_loc[1]

            if slope == 0:
                return (kart_loc[0], facing_up_grid)  # The line is horizontal, so use kart's x-coordinate.

            # Select which grid line to use based on the kart's direction.
            target_y = grid_lines_y[0] if facing_up_grid else grid_lines_y[1]
            intersection_x = (target_y - intersect_y) / slope
        except ZeroDivisionError:
            # The line is vertical, so return the x-coordinate of the kart.
            intersection_x = kart_loc[0]

        return (intersection_x, facing_up_grid)
      
    def front_flag(self, puck_loc, threshold=2.0):
      # puck_loc is expected to be a tuple or list with at least one element
      # Check if the x-coordinate is within the threshold distance from 200
      return abs(puck_loc[0] - 200) < threshold
    
    
    def model_controller(self, puck_loc, location, front, velocity, index):
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        velocity_mag = np.linalg.norm(velocity[:2])  # Assuming velocity is a 3D vector.

        # Clipping x and y values
        x, y = np.clip(puck_loc, [0, 0], [400, 300])

        if self.current_team == 'not_sure':
            self.current_team = 'red' if -58 < location[1] < -50 else 'blue'

        x_intersect, facing_up_grid = self.x_intersect(location, front)
        lean_val = 0 if -10 < location[0] < 10 else 2

        if ((facing_up_grid and 9 < x_intersect < 40) or
            (not facing_up_grid and 0 < x_intersect < 10)):
            x += lean_val if self.current_team == 'red' else -lean_val
        elif ((facing_up_grid and -40 < x_intersect < -9) or
              (not facing_up_grid and -10 < x_intersect < 0)):
            x -= lean_val if self.current_team == 'red' else lean_val

        if velocity_mag > 20:
            action['acceleration'] = 0.2

        action['steer'] = np.sign(200 - x) if x != 200 else 0
        action['drift'] = x < 50 or x > 350
        action['acceleration'] = 0.2 if action['drift'] else action['acceleration']
        action['acceleration'] = 0.5 if x < 100 or x > 300 else action['acceleration']

        # Recovery logic
        if self.recovery.get(index):
            action['steer'] = self.rescue_steer[index]
            action['acceleration'] = 0
            action['brake'] = True
            self.rescue_count[index] -= 2

            if self.rescue_count[index] < 1 or ((-57 < location[1] < 57 and -7 < location[0] < 1) and velocity_mag < 5):
                self.rescue_count[index] = 0
                self.recovery[index] = False
        else:
            if self.prev_loc[index][0] == np.int32(location)[0] and self.prev_loc[index][1] == np.int32(location)[1]:
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
        self.prev_loc[index] = np.int32(location)
        return action

    def get_instance_coords(instance, target_object=8):
      for i in range(300):
          for j in range(400):
              if instance[i][j] == target_object:
                  return True, i, j

      return False, -1, -1

    def handle_planner(self,image1, image2, player_state):
          image1 = TF.to_tensor(player_state[0])[None]
          image2 = TF.to_tensor(player_state[1])[None]
          
          #aim_point_image_Player1, _ = self.Planner(image1)
          #aim_point_image_Player2, _ = self.Planner(image2)

          if self.frame >= 30:  #call the planner
            
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            

            
            aim_point_image_Player1 = self.Planner(image1)
            aim_point_image_Player2 = self.Planner(image2)
            aim_point_image_Player1 = aim_point_image_Player1.squeeze(0)
            aim_point_image_Player2 = aim_point_image_Player2.squeeze(0)
            
          if self.frame < 30:   #do not call planner, soccer cord is likely [0,0]
            
            proj1 = np.array(player_state[0]['camera']['projection']).T
            view1 = np.array(player_state[0]['camera']['view']).T
            proj2 = np.array(player_state[1]['camera']['projection']).T
            view2 = np.array(player_state[1]['camera']['view']).T
            x = np.float32([0,0,0]) 
            aim_point_image_Player1 = self.to_screen_coordinates(x, proj1, view1) 
            aim_point_image_Player2 = self.to_screen_coordinates(x, proj2, view2)
      
    def extract_kart_data(self,player):
        Kart_A_front = player['kart']['front']
        Kart_A_location = player['kart']['location']
        Kart_A_vel = player['kart']['velocity']
        pos_A = self.to_numpy(Kart_A_location)
        front_A =self.to_numpy(Kart_A_front)
        return 
      
      
        
    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """

       
        self.my_team = player_state[0]['kart']['player_id'] % 2


        # Planner logic
        if self.planner:
            image1, image2 = [TF.to_tensor(img)[None] for img in player_image]
            aim_point_image_Player1, aim_point_image_Player2 = self.handle_planner(image1, image2, player_state) # Implement this function separately

        # Update frame and possibly debug stats
        self.frame += 1

        # Final actions
        action_A = self.model_controller(aim_point_image_Player1, *self.extract_kart_data(player_state[0]))
        action_B = self.model_controller(aim_point_image_Player2, *self.extract_kart_data(player_state[1]))

        return [action_A, action_B]

