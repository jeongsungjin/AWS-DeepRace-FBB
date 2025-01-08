import math

def reward_function(params):
    # 입력 파라미터 읽기
    speed = params['speed']
    progress = params['progress']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']

    # 기본 보상 설정
    reward = 1.0

    reward += speed

    if 0 < progress < 10:
        reward += progress * 0.2
    elif 10 <= progress < 25:
        reward += progress * 0.175
    elif 25 <= progress <= 50:
        reward += progress * 0.15
    elif 50 <= progress < 75:
        reward += progress * 0.125
    else:
        reward += progress * 0.1

    # 중심선 유지 유도
    marker = 0.3 * track_width  # 중심선 기준 마커
    if distance_from_center <= marker:
        reward += 1.0  # 중심선에 가까우면 높은 보상
    else:
        reward *= 0.5  # 중심선에서 멀어지면 보상 감소

    if not all_wheels_on_track:
        reward = 1e-3  # 트랙 이탈 시 낮은 보상

    return float(reward)