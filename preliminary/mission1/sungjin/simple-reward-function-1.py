import math

def reward_function(params):
    reward = 1.0

    # 트랙 위에 있는지 확인
    if not params["all_wheels_on_track"]:
        return 1e-3  # 크게 페널티

    # 중앙에서 얼마나 떨어져 있는가?
    distance_from_center = params["distance_from_center"]
    track_width = params["track_width"]
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    if distance_from_center <= marker_1:
        reward = 1.5
    elif distance_from_center <= marker_2:
        reward = 1.0
    elif distance_from_center <= marker_3:
        reward = 0.5
    else:
        reward = 1e-3  # 너무 많이 벗어나면 큰 페널티
    
    # 속도가 적정 범위일 때 추가 보상
    SPEED_THRESHOLD = 2.0
    if params["speed"] >= SPEED_THRESHOLD:
        reward += 0.2

    return float(reward)
