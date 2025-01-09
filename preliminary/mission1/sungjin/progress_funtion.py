def reward_function(params):
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    progress = params['progress']
    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    
    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        d_reward = 1.0
    elif distance_from_center <= marker_2:
        d_reward = 0.5
    elif distance_from_center <= marker_3:
        d_reward = 0.1
    else:
        return 1e-3  # likely crashed/ close to off track
    
    p_reward = progress * 0.2
    
    reward = d_reward + p_reward
    
    return float(reward)