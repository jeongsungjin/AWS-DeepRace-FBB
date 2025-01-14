def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    progress = params['progress']
    steering_angle = params['steering_angle'] #음수가 
    speed = params['speed']
    is_offtrack = params['is_offtrack']
    steps = params['steps']
    # closest_waypoints = params['closest_waypoints']

    if is_offtrack:
        return float(1e-3)
    
    # 중앙선 따라가기
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    
    if distance_from_center <= marker_1:
        reward_center = 1.0
    elif distance_from_center <= marker_2:
        reward_center = 0.5
    elif distance_from_center <= marker_3:
        reward_center = 0.1
    else:
        reward_center = 1e-3

    # 진행률에 따른 보상
    reward_progress = progress/100

    # 스텝당 진행률에 따른 보상
    reward_progress_per_steps = progress/steps

    # 각,스피드
    # acc_waypoints = [[0,7], [103,110]]
    reward_speed = 1e-3
    if abs(steering_angle) < 10: # 0도
        # if closest_waypoints[0] in list(range(acc_waypoints[0])) or closest_waypoints[0] in list(range(acc_waypoints[1])):
        if 3.5 < speed:
            reward_speed = 1.0
    elif abs(steering_angle) < 20: # 15도
        if 1.8 < speed < 3.0:
            reward_speed = 1.0
    else: # 30도
        if speed < 2.0:
            reward_speed = 1.0

    reward = (1.5*reward_center) + (0.5*reward_progress_per_steps) + (0.5*reward_progress) + (1.0*reward_speed)
    # 각 리워드 보상은 학습마다 가중치를 미세조정함.
    # 첫번째학습 2, 0.5, 1, 1
    # 두번째학습 2.2, 0, 1, 1
    # 세번째학습 1.5, 0.5, 0.5 ,1
    # progress_per_steps는 적은 steps 안에 많이 진행하면 보상을 주는건데 학습 초에 가중치가 높으면 자살하려는 경향이 생길까봐 가중치를 나중에 높임.
    
    return float(reward)
