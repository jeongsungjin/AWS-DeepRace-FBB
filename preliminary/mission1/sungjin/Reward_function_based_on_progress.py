def func_0_15(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']

    reward = 1e-3

    if not all_wheels_on_track:
        return 1e-4  # 트랙 이탈 시 최소 보상

    if speed > 2.0:
        reward += 1.0
    elif speed > 1.5:
        reward += 0.75
    elif speed > 1.0:
        reward += 0.5
    else:
        reward += 0.01

    if distance_from_center < 0.25 * track_width:
        reward += 1.0
    else:
        reward += 0.01  # 바깥쪽은 보상 감소

    return float(reward)


def func_15_19(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']
    steering_angle = params['steering_angle']
    is_left_of_center = params['is_left_of_center']

    reward = 1e-3

    if steering_angle < 0:
        reward += 1.0
    else:
        reward += 0.01

    if is_left_of_center == True:
        reward += 1.0
    else:
        reward += 0.01

    return float(reward)

def func_20_32(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']
    steering_angle = params['steering_angle']
    is_left_of_center = params['is_left_of_center']

    reward = 1e-3

    if -10 <= steering_angle <= 10:
        reward += 0.67
    else:
        reward += 0.01

    if speed > 1.5:
        reward += 0.67
    elif speed > 1.0:
        reward += 0.2
    else:
        reward += 0.01

    if distance_from_center < 0.35 * track_width:
        reward += 0.67
    else:
        reward += 0.01  # 바깥쪽은 보상 감소

    return float(reward)

def func_32_43(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']
    steering_angle = params['steering_angle']
    is_left_of_center = params['is_left_of_center']

    reward = 1e-3

    if -10 <= steering_angle <= 10:
        reward += 0.67
    else:
        reward += 0.01

    if speed > 2.0:
        reward += 0.67
    elif speed > 1.5:
        reward += 0.2
    else:
        reward += 0.01
    
    if distance_from_center < 0.25 * track_width:
        reward += 0.67
    else:
        reward += 0.01  # 바깥쪽은 보상 감소

    return float(reward)

def func_44_54(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']
    steering_angle = params['steering_angle']
    is_left_of_center = params['is_left_of_center']

    reward = 1e-3

    if steering_angle < 0:
        reward += 1.0
    else:
        reward += 0.01

    if is_left_of_center == True:
        reward += 1.0
    else:
        reward += 0.01

    return float(reward)


def func_55_70(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']
    steering_angle = params['steering_angle']
    is_left_of_center = params['is_left_of_center']

    reward = 1e-3

    if steering_angle > 0:
        reward += 1.0
    else:
        reward += 0.01

    if is_left_of_center == False:
        reward += 1.0
    else:
        reward += 0.01

    return float(reward)

def func_71_78(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']
    steering_angle = params['steering_angle']
    is_left_of_center = params['is_left_of_center']

    reward = 1e-3


    if steering_angle < 0:
        reward += 1.0
    else:
        reward += 0.01

    if is_left_of_center == True:
        reward += 1.0
    else:
        reward += 0.01

    return float(reward)

def func_79_85(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']
    steering_angle = params['steering_angle']
    is_left_of_center = params['is_left_of_center']

    reward = 1e-3

    if -10 <= steering_angle <= 10:
        reward += 0.67
    else:
        reward += 0.01

    if speed > 1.3:
        reward += 0.67
    else:
        reward += 0.01
    
    if distance_from_center < 0.35 * track_width:
        reward += 0.67
    else:
        reward += 0.01  # 바깥쪽은 보상 감소

    return float(reward)

def func_86_94(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']
    steering_angle = params['steering_angle']
    is_left_of_center = params['is_left_of_center']

    reward = 1e-3

    if steering_angle < 0:
        reward += 1.0
    else:
        reward += 0.01

    if is_left_of_center == True:
        reward += 1.0
    else:
        reward += 0.01

    return float(reward)

def func_94_110(params):
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']

    reward = 1e-3

    if not all_wheels_on_track:
        return 1e-4  # 트랙 이탈 시 최소 보상

    if speed > 2.0:
        reward += 1.0
    elif speed > 1.5:
        reward += 0.75
    elif speed > 1.0:
        reward += 0.5
    else:
        reward += 0.01

    if distance_from_center < 0.25 * track_width:
        reward += 1.0
    else:
        reward += 0.01  # 바깥쪽은 보상 감소

    return float(reward)

def reward_function(params):
    
    distance_from_center = params['distance_from_center']
    progress = params['progress']
    is_left_of_center = params['is_left_of_center']
    track_width = params['track_width']
    
    reward = 1e-3  # 기본 보상
    
    if 0 <= progress <= 11.5:
        reward = func_0_15(params)
    elif 11.6 <= progress <= 16.1:
        reward = func_15_19(params)
    elif 16.2 <= progress <= 28.0:
        reward = func_20_32(params)
    elif 28.1 <= progress <= 37.9:
        reward = func_32_43(params)
    elif 38.0 <= progress <= 48.1:
        reward = func_44_54(params)
    elif 48.2 <= progress <= 61.9:
        reward = func_55_70(params)
    elif 62.0 <= progress <= 69.6:
        reward = func_71_78(params)
    elif 69.7 <= progress <= 75.9:
        reward = func_79_85(params)
    elif 76.0 <= progress <= 84.3:
        reward = func_86_94(params)
    else:
        reward = func_94_110(params)
    
    return float(reward)
