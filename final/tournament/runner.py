import os
import logging
import numpy as np
from collections import namedtuple
import pystk
from os import makedirs

from argparse import ArgumentParser
from pathlib import Path
from os import environ
from . import remote, utils

TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000
TRACK_OFFSET = 15

TIMEOUT_SLACK = 2   # seconds
TIMEOUT_STEP = 0.1  # seconds

DATASET_PATH = 'drive_data'
ON_COLAB = os.environ.get('ON_COLAB', False)
COLAB_IMAGES = list()

print(ON_COLAB)



RunnerInfo = namedtuple('RunnerInfo', ['agent_type', 'error', 'total_act_time'])



def show_on_colab():
    from moviepy.editor import ImageSequenceClip
    from IPython.display import display

    display(ImageSequenceClip(COLAB_IMAGES, fps=15).ipython_display(width=512, autoplay=True, loop=True, maxduration=120))

def to_native(o):
    # Super obnoxious way to hide pystk
    import pystk
    _type_map = {pystk.Camera.Mode: int,
                 pystk.Attachment.Type: int,
                 pystk.Powerup.Type: int,
                 float: float,
                 int: int,
                 list: list,
                 bool: bool,
                 str: str,
                 memoryview: np.array,
                 property: lambda x: None}

    def _to(v):
        if type(v) in _type_map:
            return _type_map[type(v)](v)
        else:
            return {k: _to(getattr(v, k)) for k in dir(v) if k[0] != '_'}
    return _to(o)


class AIRunner:
    agent_type = 'image'
    is_ai = True
    
    @staticmethod
    def _g(f):
        from .remote import ray
        if ray is not None and isinstance(f, (ray.types.ObjectRef, ray._raylet.ObjectRef)):
            return ray.get(f)
        return f

    def new_match(self, team: int, num_players: int) -> list:
        pass

    def act(self, player_state, opponent_state, world_state):
        return []

    def info(self):
        return RunnerInfo('state', None, 0)


class TeamRunner:
    agent_type = 'state'
    _error = None
    _total_act_time = 0

    def __init__(self, team_or_dir):
        from pathlib import Path
        try:
            from grader import grader
        except ImportError:
            try:
                from . import grader
            except ImportError:
                import grader

        self._error = None
        self._team = None
        try:
            if isinstance(team_or_dir, (str, Path)):
                assignment = grader.load_assignment(team_or_dir)
                if assignment is None:
                    self._error = 'Failed to load submission.'
                else:
                    self._team = assignment.Team()
            else:
                self._team = team_or_dir
        except Exception as e:
            self._error = 'Failed to load submission: {}'.format(str(e))
        if hasattr(self, '_team') and self._team is not None:
            self.agent_type = self._team.agent_type

    def new_match(self, team: int, num_players: int) -> list:
        self._total_act_time = 0
        self._error = None
        try:
            r = self._team.new_match(team, num_players)
            if isinstance(r, str) or isinstance(r, list) or r is None:
                return r
            self._error = 'new_match needs to return kart names as a str, list, or None. Got {!r}!'.format(r)
        except Exception as e:
            self._error = 'Failed to start new_match: {}'.format(str(e))
        return []

    def act(self, player_state, *args, **kwargs):
        from time import time
        t0 = time()
        try:
            r = self._team.act(player_state, *args, **kwargs)
        except Exception as e:
            self._error = 'Failed to act: {}'.format(str(e))
        else:
            self._total_act_time += time()-t0
            return r
        return []

    def info(self):
        return RunnerInfo(self.agent_type, self._error, self._total_act_time)


class MatchException(Exception):
    def __init__(self, score, msg1, msg2):
        self.score, self.msg1, self.msg2 = score, msg1, msg2


class Match:
    """
        Do not create more than one match per process (use ray to create more)
    """
    
    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)
    _singleton = None
    
    def _to_image300_400(self, x, proj, view):
        W, H = 400, 300
        op = np.array(list(x) + [1])  # Convert x to homogeneous coordinates
        p = proj @ view @ op  # Matrix multiplication

        # Combine transformations and scaling into a single step
        aimpoint = np.array([(W / 2) * (p[0] / p[-1] + 1), 
                            (H / 2) * (-p[1] / p[-1] + 1)])

        return aimpoint

    def collect(_, im, puck_flag, pt, instance=None):
        global file_no 
        id = file_no 
        divide_data = False  
        save_data = True     
        instance_data = False  

        if save_data:
            # Define base directory based on puck_flag and divide_data
            if puck_flag and divide_data:
                base_dir = '/content/cs342/final/data_YesPuck/'
            elif not puck_flag and divide_data:
                base_dir = '/content/cs342/final/data_NoPuck/'
            else:
                base_dir = '/content/cs342/final/data_instance/' if instance_data else '/content/cs342/final/data/'

            fn = path.join(base_dir, 'ice_hockey' + '_%05d' % id)
            Image.fromarray(im).save(fn + '.png')

            # Save additional data based on instance_data flag
            if instance_data:
                # Image.fromarray(instance).save(fn + '_instance' + '.png')
                # torch.save(instance, fn + '_instance' + '_tensor.pt')
                with open(fn + '.npy', 'wb') as f:
                    np.save(f, instance)
            else:
                with open(fn + '.csv', 'w') as f:
                    # f.write('%0.1f,%0.1f,%0.1f' % (pt[0], pt[1], puck_flag))  # with puck flag
                    f.write('%0.1f,%0.1f' % tuple(pt))

            file_no += 1
    
    
    def __init__(self, use_graphics=True, logging_level=None):
        # DO this here so things work out with ray
        import pystk
        self._pystk = pystk
        if logging_level is not None:
            logging.basicConfig(level=logging_level)

        # Fire up pystk
        self._use_graphics = use_graphics
        if use_graphics:
            graphics_config = self._pystk.GraphicsConfig.hd()
            graphics_config.screen_width = 400
            graphics_config.screen_height = 300
        else:
            graphics_config = self._pystk.GraphicsConfig.none()

        self._pystk.init(graphics_config)

    def __del__(self):
        if hasattr(self, '_pystk') and self._pystk is not None and self._pystk.clean is not None:  # Don't ask why...
            self._pystk.clean()

    def _make_config(self, team_id, is_ai, kart):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    @classmethod
    def _r(cls, f):
        if hasattr(f, 'remote'):
            return f.remote
        if hasattr(f, '__call__'):
            if hasattr(f.__call__, 'remote'):
                return f.__call__.remote
        return f

    @staticmethod
    def _g(f):
        from .remote import ray
        if ray is not None and isinstance(f, (ray.types.ObjectRef, ray._raylet.ObjectRef)):
            return ray.get(f)
        return f

    def _check(self, team1, team2, where, n_iter, timeout):
        _, error, t1 = self._g(self._r(team1.info)())
        if error:
            raise MatchException([0, 3], 'other team crashed', 'crash during {}: {}'.format(where, error))

        _, error, t2 = self._g(self._r(team2.info)())
        if error:
            raise MatchException([3, 0], 'crash during {}: {}'.format(where, error), 'other team crashed')
        
       
        
        logging.debug('timeout {} <? {} {}'.format(timeout, t1, t2))
        return t1 < timeout[0], t2 < timeout[1]

    @classmethod
    def run(self, team1, team2, num_player=1, max_frames=MAX_FRAMES, max_score=3, record_fn=None, timeout=1e10,
            initial_ball_location=[0, 0], initial_ball_velocity=[0, 0], verbose=False):

        timeout=TIMEOUT_SLACK
        verbose= True
        logging.info('RUN')
        
        if verbose and ON_COLAB:
            global COLAB_IMAGES
            COLAB_IMAGES = list()

        # Start a new match
        t1_cars = self._g(self._r(team1.new_match)(0, num_player)) or ['tux']
        t2_cars = self._g(self._r(team2.new_match)(1, num_player)) or ['tux']

        t1_type, *_ = self._g(self._r(team1.info)())
        t2_type, *_ = self._g(self._r(team2.info)())

        if t1_type == 'image' or t2_type == 'image':
            assert self._use_graphics, 'Need to use_graphics for image agents.'

        # Deal with crashes
        t1_can_act, t2_can_act = self._check(team1, team2, 'new_match', 0, TIMEOUT_SLACK, TIMEOUT_STEP)

        # Setup the race config
        logging.info('Setting up race')

        race_config = self._pystk.RaceConfig(track=TRACK_NAME, mode=self._pystk.RaceConfig.RaceMode.SOCCER, num_kart=2 * num_player)
        race_config.players.pop()
        for i in range(num_player):
            race_config.players.append(self._make_config(0, hasattr(team1, 'is_ai') and team1.is_ai, t1_cars[i % len(t1_cars)]))
            race_config.players.append(self._make_config(1, hasattr(team2, 'is_ai') and team2.is_ai, t2_cars[i % len(t2_cars)]))
        
        # Start the match
        logging.info('Starting race')
      
        if(self.isRaceRunning == False):
            race = self._pystk.Race(race_config)
            race.start()
            self.isRaceRunning = True
            race.step()
        

        state = self._pystk.WorldState()
        state.update()
        print(initial_ball_location)
        state.set_ball_location((initial_ball_location[0], 1, initial_ball_location[1]),
                                (initial_ball_velocity[0], 0, initial_ball_velocity[1]))
        
        

        for it in range(max_frames):
            logging.debug('iteration {} / {}'.format(it, MAX_FRAMES))
            state.update()

            # Get the state
            team1_state = [to_native(p) for p in state.players[0::2]]
            team2_state = [to_native(p) for p in state.players[1::2]]
            soccer_state = to_native(state.soccer)
            team1_images = team2_images = None
            if self._use_graphics:
                team1_images = [np.array(race.render_data[i].image) for i in range(0, len(race.render_data), 2)]
                team2_images = [np.array(race.render_data[i].image) for i in range(1, len(race.render_data), 2)]
                heatmap_team1 = [race.render_data[i].instance for i in range(0, len(race.render_data), 2)]
                #heatmap_team2 = [race.render_data[i].instance for i in range(1, len(race.render_data), 2)]

            # Have each team produce actions (in parallel)
            if t1_can_act:
                if t1_type == 'image':
                    team1_actions_delayed = self._r(team1.act)(team1_state, team1_images)
                else:
                    team1_actions_delayed = self._r(team1.act)(team1_state, team2_state, soccer_state)

            if t2_can_act:
                if t2_type == 'image':
                    team2_actions_delayed = self._r(team2.act)(team2_state, team2_images)
                else:
                    team2_actions_delayed = self._r(team2.act)(team2_state, team1_state, soccer_state)

            # Wait for the actions to finish
            team1_actions = self._g(team1_actions_delayed) if t1_can_act else None
            team2_actions = self._g(team2_actions_delayed) if t2_can_act else None

            new_t1_can_act, new_t2_can_act = self._check(team1, team2, 'act', it, timeout)
            if not new_t1_can_act and t1_can_act and verbose:
                print('Team 1 timed out')
            if not new_t2_can_act and t2_can_act and verbose:
                print('Team 2 timed out')

            t1_can_act, t2_can_act = new_t1_can_act, new_t2_can_act

            # Assemble the actions
            actions = []
            for i in range(num_player):
                a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
                a2 = team2_actions[i] if team2_actions is not None and i < len(team2_actions) else {}
                actions.append(a1)
                actions.append(a2)
                
            print('players')
            print(state.players)
            

            x=soccer_state['ball']['location'][0]
            y = soccer_state['ball']['location'][1] 
            z=soccer_state['ball']['location'][2]
            xyz = np.random.rand(3)
            xz=np.random.rand(2)
            xz[0] = x
            xz[1] = z
            xyz[0] = x
            xyz[1] = y
            xyz[2] = z
            
            aim_point_300_400 = np.clip(aim_point_300_400, [0, 0], [400, 300])
            aim_point_300_400_2 = np.clip(aim_point_300_400_2, [0, 0], [400, 300])
            
            proj = np.array(team1_state[0]['camera']['projection']).T
            view = np.array(team1_state[0]['camera']['view']).T
            
            proj2 = np.array(team2_state[0]['camera']['projection']).T
            view2 = np.array(team2_state[0]['camera']['view']).T

            #aim_point_world = self._point_on_track(kart.distance_down_track+TRACK_OFFSET, TRACK_NAME)
            aim_point_image, out_of_frame = self._to_image(xyz, proj, view)  
            #aim_point_image2, out_of_frame = self._to_image(xyz, proj2, view2) 
            
            if heatmap_team1:
                # Right shift the entire array at once
                heatmap_team1[0] >>= 24

                # Use numpy to check if any element equals 8
                puck_flag = int(np.any(heatmap_team1[0] == 8))

            if record_fn:
                self._r(record_fn)(team1_state, team2_state, soccer_state=soccer_state, actions=actions,
                                   team1_images=team1_images, team2_images=team2_images)
                self.collect(team1_images[0], puck_flag, aim_point_image)

                if verbose and ON_COLAB:
                    from PIL import Image, ImageDraw
                    image = Image.fromarray(self.k.render_data[0].image)
                    draw = ImageDraw.Draw(image)

                    WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2

                    p = (aim_point_image + 1) * WH2
                    draw.ellipse((p[0] - 2, p[1] - 2, p[0]+2, p[1]+2), fill=(255, 0, 0))

                    COLAB_IMAGES.append(np.array(team1_images[0]))

            logging.debug('  race.step  [score = {}]'.format(state.soccer.score))
            if (not race.step([self._pystk.Action(**a) for a in actions]) and num_player) or sum(state.soccer.score) >= max_score:
                break

        if verbose and ON_COLAB:
            show_on_colab()
            
        race.stop()
        del race

        return state.soccer.score

    def wait(self, x):
        return x
    

#-----------------------------------------------------------
if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    from os import environ
    from . import remote, utils
    from . import runner

    parser = ArgumentParser(description="Play some Ice Hockey. List any number of players, odd players are in team 1, even players team 2.")
    parser.add_argument('-r', '--record_video', help="Do you want to record a video?")
    parser.add_argument('-s', '--record_state', help="Do you want to pickle the state?")
    parser.add_argument('-f', '--num_frames', default=1200, type=int, help="How many steps should we play for?")
    parser.add_argument('-p', '--num_players', default=2, type=int, help="Number of players per team")
    parser.add_argument('-m', '--max_score', default=3, type=int, help="How many goal should we play to?")
    parser.add_argument('-j', '--parallel', type=int, help="How many parallel process to use?")
    parser.add_argument('--ball_location', default=[0, 0], type=float, nargs=2, help="Initial xy location of ball")
    parser.add_argument('--ball_velocity', default=[0, 0], type=float, nargs=2, help="Initial xy velocity of ball")
    parser.add_argument('team1', help="Python module name or `AI` for AI players.")
    parser.add_argument('team2', help="Python module name or `AI` for AI players.")
    args = parser.parse_args()

    logging.basicConfig(level=environ.get('LOGLEVEL', 'WARNING').upper())
    


    if args.parallel is None or remote.ray is None:
        # Create the teams
        team1 = AIRunner() if args.team1 == 'AI' else TeamRunner(args.team1)
        team2 = AIRunner() if args.team2 == 'AI' else TeamRunner(args.team2)
        
        print(team1)
        print(team2)

        # What should we record?
        recorder = None
        if args.record_video:
            recorder = recorder & utils.VideoRecorder(args.record_video)

        if args.record_state:
            recorder = recorder & utils.StateRecorder(args.record_state)

        # Start the match
        #match = Match(use_graphics=team1.agent_type == 'image' or team2.agent_type == 'image')
        match = runner.Match()
        try:
            result = match.run(team1, team2, args.num_players, args.num_frames, max_score=3,
                               initial_ball_location=args.ball_location, initial_ball_velocity=args.ball_velocity,
                               record_fn=recorder)
        except MatchException as e:
            print('Match failed', e.score)
            print(' T1:', e.msg1)
            print(' T2:', e.msg2)

        print('Match results', result)

    else:
        # Fire up ray
        remote.init(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()), configure_logging=True,
                    log_to_driver=True, include_dashboard=False)

        # Create the teams
        team1 = AIRunner() if args.team1 == 'AI' else remote.RayTeamRunner.remote(args.team1)
        team2 = AIRunner() if args.team2 == 'AI' else remote.RayTeamRunner.remote(args.team2)
        team1_type, *_ = team1.info() if args.team1 == 'AI' else remote.get(team1.info.remote())
        team2_type, *_ = team2.info() if args.team2 == 'AI' else remote.get(team2.info.remote())

        # What should we record?
        assert args.record_state is None or args.record_video is None, "Cannot record both video and state in parallel mode"

        # Start the match
        results = []
        for i in range(args.parallel):
            recorder = None
            if args.record_video:
                ext = Path(args.record_video).suffix
                recorder = remote.RayVideoRecorder.remote(args.record_video.replace(ext, f'.{i}{ext}'))
            elif args.record_state:
                ext = Path(args.record_state).suffix
                recorder = remote.RayStateRecorder.remote(args.record_state.replace(ext, f'.{i}{ext}'))

            match = remote.RayMatch.remote(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()),
                                           use_graphics=team1_type == 'image' or team2_type == 'image')
            result = match.run.remote(team1, team2, args.num_players, args.num_frames, max_score=args.max_score,
                                      initial_ball_location=args.ball_location,
                                      initial_ball_velocity=args.ball_velocity,
                                      record_fn=recorder)
            
            
            results.append(result)

        for result in results:
            try:
                result = remote.get(result)
            except (remote.RayMatchException, MatchException) as e:
                print('Match failed', e.score)
                print(' T1:', e.msg1)
                print(' T2:', e.msg2)

            print('Match results', result)
            
        match.close()
