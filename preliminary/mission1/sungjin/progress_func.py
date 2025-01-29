import math

def reward_function(params):
    """
    트랙 구간별(progress) 특성 반영:
      - 0~11%: 고속 직진
      - 12~16%: 좌회전 U턴
      - 17~28%: 좌우 살짝 굴곡(거의 직진)
      - 28~37%: 약간 왼쪽 곡선(거의 직진)
      - 38~47%: 좌회전 U턴
      - 48~61%: 우회전 U턴
      - 62~69%: 왼쪽 90도 (인코스)
      - 70~75%: 좌우 굴곡(거의 직진)
      - 76~84%: 좌회전 U턴
      - 85~100%: 직진
    """

    #########################################################################
    # 1. 기본 파라미터 로드
    #########################################################################
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    steering_angle = abs(params['steering_angle'])  # 좌우 상관없이 절대값
    progress = params['progress']  # 0~100
    steps = params['steps']

    # Waypoints 및 현재 구간 방향
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    prev_wp = waypoints[closest_waypoints[0]]
    next_wp = waypoints[closest_waypoints[1]]

    heading = params['heading']

    # 기본 보상 초기값
    reward = 1.0

    #########################################################################
    # 2. 트랙 이탈 방지
    #########################################################################
    if not all_wheels_on_track:
        return 1e-3

    #########################################################################
    # 3. 트랙 중앙 거리 보상 (마커 방식)
    #########################################################################
    marker_1 = 0.2 * track_width
    marker_2 = 0.35 * track_width
    marker_3 = 0.5 * track_width

    if distance_from_center <= marker_1:
        center_factor = 1.0
    elif distance_from_center <= marker_2:
        center_factor = 0.75
    elif distance_from_center <= marker_3:
        center_factor = 0.5
    else:
        center_factor = 1e-3  # 거의 이탈 직전

    reward *= center_factor

    #########################################################################
    # 4. 코너/직선: 헤딩(heading) 차이 보상
    #########################################################################
    #   - 이 구간이 코너인지 직선인지 판단하지 않고, 기본적으로
    #     waypoint기반 '트랙 진행방향'과 heading 차이가 작을수록 추가 보상
    #########################################################################
    track_dx = next_wp[0] - prev_wp[0]
    track_dy = next_wp[1] - prev_wp[1]
    track_angle = math.degrees(math.atan2(track_dy, track_dx))

    heading_diff = abs(track_angle - heading)
    if heading_diff > 180:
        heading_diff = 360 - heading_diff

    if heading_diff <= 10:
        heading_factor = 1.0
    elif heading_diff <= 30:
        heading_factor = max(0.1, 1.0 - (heading_diff - 10)/20.0)
    else:
        heading_factor = 1e-3

    reward += heading_factor

    #########################################################################
    # 5. 속도 & 스티어링 안정성
    #########################################################################
    #  - 속도 높을수록 좋음 (단, 너무 높으면 커브에서 이탈 위험)
    #  - 스티어링 각도 너무 크면 페널티
    #########################################################################


    #########################################################################
    # 6. 구간별(progress) 특성 반영 (핵심)
    #########################################################################
    #   - 아래는 예시로 구간별로 원하는 '속도/스티어링'을 간단히 분기해
    #     추가 보상(혹은 패널티)로 반영
    #########################################################################
    p = progress

    if 0 <= p < 12:  # (0~11%)
        # 직진 구간 => 빠른 속도, 스티어링 최소
        if speed > 3.0:  # 3m/s 이상이면 +
            reward += 0.5
        if steering_angle < 5.0:  # 거의 직진
            reward += 0.5

    elif 12 <= p < 17:  # (12~16%) 좌회전 유턴
        # U턴 => 너무 빠르면 위험, 적당히 낮은 속도 + 큰 스티어링 허용
        if speed < 2.0:
            reward += 0.5
        if steering_angle > 15.0:  # 어느 정도 큰 각도 필요
            reward += 0.5

    elif 17 <= p < 29:  # (17~28%) 좌우 굴곡 but 거의 직진
        # 최소 조향, 중~고속
        if steering_angle < 5.0:
            reward += 0.5
        if speed > 2.5:
            reward += 0.3

    elif 29 <= p < 38:  # (28~37%) 왼쪽 살짝 휨(거의 직진)
        if steering_angle < 5.0:
            reward += 0.5
        if speed > 2.5:
            reward += 0.3

    elif 38 <= p < 48:  # (38~47%) 좌회전 유턴
        if speed < 2.0:
            reward += 0.5
        if steering_angle > 15.0:
            reward += 0.5

    elif 48 <= p < 62:  # (48~61%) 우회전 유턴
        if speed < 2.0:
            reward += 0.5
        if steering_angle > 15.0:
            reward += 0.5

    elif 62 <= p < 70:  # (62~69%) 왼쪽 90도 (인코스)
        # 강한 왼쪽 회전 → 속도는 낮고 스티어링은 꽤 커야할 수도
        if speed < 2.0:
            reward += 0.7
        if steering_angle > 10.0:
            reward += 0.3

    elif 70 <= p < 76:  # (70~75%) 좌우 약간 굴곡(거의 직진)
        if steering_angle < 5.0:
            reward += 0.5
        if speed > 2.5:
            reward += 0.3

    elif 76 <= p < 85:  # (76~84%) 좌회전 유턴
        if speed < 2.0:
            reward += 0.5
        if steering_angle > 15.0:
            reward += 0.5

    else:  # (85~100%) 직진 구간
        if speed > 3.0:
            reward += 0.5
        if steering_angle < 5.0:
            reward += 0.5

    #########################################################################
    # 7. 진행도(progress) 기반 - 빠른 랩 완주 유도
    #########################################################################
    #   - steps 대비 progress(%)를 이용해서 주행 효율이 높으면 보너스
    #########################################################################
    if 'steps' in params and params['steps'] > 0:
        time_seconds = params['steps'] / 15  # 스텝을 초 단위로 변환
        efficiency = progress / time_seconds  # 시간 대비 진행률 계산
        if efficiency > 5.0:  # 효율성이 높은 경우
            reward += 1.0  # 높은 추가 보상
        elif efficiency > 3.0:
            reward += 0.7  # 중간 수준 보상
        else:
            reward += 0.0  # 기본 추가 보상    reward *= time_factor

    # 완주 보너스
    if progress >= 100:
        reward += 30.0

    #########################################################################
    # 최종 반환
    #########################################################################
    return float(reward)
