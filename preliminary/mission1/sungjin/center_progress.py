def reward_function(params):
    # 입력 파라미터 읽기
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    progress = params['progress']

    # 중앙선과의 거리 기준 설정
    marker_1 = 0.1 * track_width  # 중앙선에서 10% 이내
    marker_2 = 0.25 * track_width  # 중앙선에서 25% 이내
    marker_3 = 0.5 * track_width  # 중앙선에서 50% 이내

    # 보상 초기화
    reward = 1e-3  # 기본 보상값

    # 중앙선에 가까울수록 보상 증가
    if distance_from_center <= marker_1:
        reward += 1.0  # 중앙선에서 가장 가까운 경우
    elif distance_from_center <= marker_2:
        reward += 0.5  # 중앙선에서 조금 떨어진 경우
    elif distance_from_center <= marker_3:
        reward += 0.1  # 중앙선에서 많이 떨어진 경우

    # 진행률에 따른 추가 보상
    reward += progress * 0.1  # 진행률에 비례하여 보상 증가

    return float(reward)