def reward_function(params):
    # 주요 파라미터
    speed = params['speed']  # 차량 속도 (m/s)
    progress = params['progress']  # 진행률 (%)
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    progress = params['progress']

    
    if not params['all_wheels_on_track']:
        return 1e-4  # 트랙 이탈 시 최소 보상

    # 보상 초기화
    reward = 0.0

    # 최적 속도 범위 설정
    OPTIMAL_SPEED_MIN = 2.0  # 최소 최적 속도 (m/s)
    OPTIMAL_SPEED_MAX = 4.0  # 최대 최적 속도 (m/s)

    # 속도에 대한 보상
    if speed < OPTIMAL_SPEED_MIN:
        speed_reward = 0.1  # 속도가 최적 이하일 때 낮은 보상
    elif speed > OPTIMAL_SPEED_MAX:
        speed_reward = 0.1  # 속도가 최적 이상일 때 낮은 보상
    else:
        # 최적 속도 범위 내에서 보상 계산
        speed_reward = (speed - OPTIMAL_SPEED_MIN) / (OPTIMAL_SPEED_MAX - OPTIMAL_SPEED_MIN)
    # 중앙선과의 거리 기준 설정
    marker_1 = 0.25 * track_width  # 중앙선에서 10% 이내
    marker_2 = 0.5 * track_width  # 중앙선에서 25% 이내
    marker_3 = 0.7 * track_width  # 중앙선에서 50% 이내

    
    # 중앙선에 가까울수록 보상 증가
    if distance_from_center <= marker_1:
        reward += 1.0  # 중앙선에서 가장 가까운 경우
    elif distance_from_center <= marker_2:
        reward += 0.75  # 중앙선에서 조금 떨어진 경우
    elif distance_from_center <= marker_3:
        reward += 0.1  # 중앙선에서 많이 떨어진 경우

    progress_reward = progress / 70.0

    # 종합 보상 계산
    reward = speed_reward + progress_reward

    # 보상 값의 범위 제한 (최소 0.0, 최대 2.0)
    reward = max(0.0, min(reward, 4.0))

    return float(reward)