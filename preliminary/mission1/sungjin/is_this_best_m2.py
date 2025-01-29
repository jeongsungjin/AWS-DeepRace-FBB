def reward_function(params):
    
    distance_from_center = params['distance_from_center']
    progress = params['progress']
    is_left_of_center = params['is_left_of_center']
    track_width = params['track_width']
    
    reward = 1e-3  # 기본 보상
    
    if not params['all_wheels_on_track']:
        return 1e-4  # 트랙 이탈 시 최소 보상
    
    # 트랙 중앙 유지 보상
    if distance_from_center < 0.35 * track_width:
        reward += 1.2
    else:
        reward += 0.1  # 바깥쪽은 보상 감소
    
    # 진행률 기반 보상 세분화, 이거 필요한지?
    if progress >= 90:  # 트랙의 마지막 10%
        reward += 2.0  # 완주를 유도하는 높은 보상
    elif progress >= 50:
        reward += 1.2  # 중간 구간에서는 안정적 진행 유도
    else:
        reward += 0.5  # 초기 구간에서는 상대적으로 낮은 보상
    
    if 45 <= progress <= 55:
        if not is_left_of_center:
            reward += 1.0
    else:
        if is_left_of_center:
            reward += 1.0
    
    if 'steps' in params and params['steps'] > 0:
        time_seconds = params['steps'] / 15  # 스텝을 초 단위로 변환
        efficiency = progress / time_seconds  # 시간 대비 진행률 계산
        if efficiency > 9.0:  # 효율성이 높은 경우
            reward += 1.5  # 높은 추가 보상
        elif efficiency > 6.0:
            reward += 1.0  # 중간 수준 보상
        else:
            reward += 0.0  # 기본 추가 보상
    
    return float(reward)
