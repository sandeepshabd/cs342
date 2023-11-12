
import pystk
import numpy as np

STEER_GAIN = 1.0  # This needs to be tuned
VELOCITY_GAIN = 0.1  # This needs to be tuned
BRAKE_THRESHOLD = 0.1  # This needs to be tuned
MAX_VELOCITY = 5.0  # Set this to the desired max velocity

def calculate_steering_angle(x):
    # Simple proportional control
    return x

def calculate_acceleration(current_velocity):
    # Implement your logic here
    # For example, full acceleration if not at max speed
    return 1.0 if current_velocity < MAX_VELOCITY else 0.5

def should_brake(aim_point, current_velocity):
    # Implement your logic here
    # For example, brake if the turn is sharp and speed is high
    return abs(aim_point.x) > 0.5 and current_velocity > MAX_VELOCITY

def should_drift(aim_point):
    # Drift on sharp turns
    return abs(aim_point.x) > 0.7

def should_use_nitro(current_velocity, aim_point):
    # Use nitro in certain conditions
    return current_velocity < MAX_VELOCITY and abs(aim_point.x) < 0.3


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)

    # Calculate the steering angle
    # Assuming aim_point.x and aim_point.y are normalized between -1 and 1
    action.steer = calculate_steering_angle(aim_point.x)

    # Set acceleration and braking
    action.acceleration = calculate_acceleration(current_velocity)
    action.brake = should_brake(aim_point, current_velocity)

    # Determine if drifting or nitro should be used
    action.drift = should_drift(aim_point)
    action.nitro = should_use_nitro(current_velocity)
    
    
    """
    
     # Calculate the desired steering angle
    # aim_point[0] is the x-coordinate of the aim point (-1 to 1)
    # We invert the x value since the coordinate system is left-handed
    steering_angle = -aim_point[0] * STEER_GAIN
    action.steer = np.clip(steering_angle, -1.0, 1.0)
    
    # Calculate the desired acceleration
    current_speed = np.linalg.norm(current_vel)
    if current_speed < MAX_VELOCITY:
        # If below max velocity, continue accelerating
        action.acceleration = 1.0
    else:
        # Otherwise, coast or brake
        action.acceleration = 0.0
    
    # Decide whether to brake
    if np.abs(steering_angle) > BRAKE_THRESHOLD and current_speed > MAX_VELOCITY:
        action.brake = True
    else:
        action.brake = False
    
    # Decide whether to use nitro
    action.nitro = current_speed < MAX_VELOCITY and np.abs(steering_angle) < BRAKE_THRESHOLD
    
    # Decide whether to drift
    # Drift if the aim point is far to the side and the speed is high
    action.drift = np.abs(steering_angle) > BRAKE_THRESHOLD and current_speed > MAX_VELOCITY

    return action

def test_controller(args):
    import numpy as np
    pytux = PyTux()
    for t in args.track:
        steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
        print(steps, how_far)
    pytux.close()

if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser




    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
