import math

# 웨이포인트 데이터
WAYPOINTS = [
    [0.70823, 0.35672, 4.0],  # [x, y, speed]
    [0.50682, 0.58334, 4.0],
    [0.30522, 0.80979, 4.0],
    [0.10341, 1.03605, 4.0],
    [-0.09859, 1.26214, 4.0],
    # 이후 웨이포인트는 생략
]

def reward_function(params):
    # 현재 차량 상태 불러오기
    car_x = params['x']
    car_y = params['y']
    car_speed = params['speed']
    car_heading = params['heading']

    # 트랙 안에 있는지 확인
    if not params['all_wheels_on_track']:
        return 1e-3  # 트랙 이탈 시 최소 보상

    # 가장 가까운 웨이포인트 찾기
    nearest_idx = find_nearest_waypoint(car_x, car_y)

    # 두 개 앞 웨이포인트 인덱스 계산
    target_idx = (nearest_idx + 2) % len(WAYPOINTS)
    
    # 목표 웨이포인트 데이터 가져오기
    target_wp = WAYPOINTS[target_idx]
    target_x, target_y, target_speed = target_wp
    
    # 현재 웨이포인트에서 목표 웨이포인트까지의 방향(헤딩) 계산
    target_heading = math.degrees(math.atan2(target_y - car_y, target_x - car_x))
    
    # 현재 헤딩과 목표 헤딩 차이 계산 (차량이 향하는 방향 조절)
    heading_difference = abs(target_heading - car_heading)
    heading_difference = min(heading_difference, 360 - heading_difference)  # 0~180도로 제한
    
    # 속도 오차 계산
    speed_difference = abs(target_speed - car_speed)

    # 보상 계산
    reward = 1.0  # 기본 보상
    
    # 헤딩 차이가 작을수록 높은 보상 (최대 보상 1, 오차 클수록 0에 가까움)
    heading_reward = max(0.0, 1 - (heading_difference / 30.0))  # 30도 이상이면 페널티
    reward += heading_reward * 2  # 헤딩 보상 가중치 2배
    
    # 속도 차이가 작을수록 높은 보상
    speed_reward = max(0.0, 1 - (speed_difference / target_speed))  # 속도 오차 보상
    reward += speed_reward * 1.5  # 속도 보상 가중치 1.5배
    
    return float(reward)


# 현재 위치에서 가장 가까운 웨이포인트 찾기
def find_nearest_waypoint(car_x, car_y):
    min_dist = float("inf")
    nearest_idx = 0
    for i, (x, y, _) in enumerate(WAYPOINTS):
        dist = math.sqrt((car_x - x)**2 + (car_y - y)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    return nearest_idx
