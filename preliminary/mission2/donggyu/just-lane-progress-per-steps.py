CORNERS_RIGHT = [[19,27],[55,80],[122,137]]
CRITICAL_DISTANCE = 0.5

def reward_function(params):
    is_offtrack = params['is_offtrack']
    is_crashed = params['is_crashed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    objects_distance = params['objects_distance']
    _, next_object_index = params['closest_objects']
    objects_left_of_center = params['objects_left_of_center']
    is_left_of_center = params['is_left_of_center']
    _, closest_waypoint = params['closest_waypoints']
    progress = params['progress']
    steps = params['steps']

    if is_offtrack or is_crashed:
        return 1e-3
    
    # default reward
    reward = 1e-3

    # reward_avoid = 장애물 에 너무근접하면 1e-3 보상
    distance_closest_object = objects_distance[next_object_index]
    is_same_lane = objects_left_of_center[next_object_index] == is_left_of_center

    if is_same_lane and distance_closest_object < CRITICAL_DISTANCE:
        reward_avoid = 1e-3
    else:
        reward_avoid = 1.0

    # reward_driving = 주행보상
    should_left_lane = True
    for corner_right in CORNERS_RIGHT:
        if closest_waypoint in range(corner_right[0], corner_right[1]):
            should_left_lane = False

    if (0.5 * track_width - distance_from_center) >= 0:
        reward_lane = 1.0
        if should_left_lane==is_left_of_center:
            reward_lane = 1.5
    else:
        reward_lane = 1e-3

    reward_progress_per_steps = progress/steps

    # 보상에서 reward_progress_per_steps는 학습이 진행됨에따라 점점 올
    reward = 1.0 * reward_avoid + 3.0 * reward_lane + 1.2 * reward_progress_per_steps

    return reward
