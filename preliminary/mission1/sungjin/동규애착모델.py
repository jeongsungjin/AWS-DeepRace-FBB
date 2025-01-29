import math

# 최적화된 waypoint들 (x, y, speed)
waypoints_opt = [[0.25792, 0.86332, 4.0],
                [0.11893, 1.01891, 4.0],
                [-0.01874, 1.17284, 4.0],
                [-0.17592, 1.34867, 4.0],
                [-0.34798, 1.54151, 4.0],
                [-0.52995, 1.74581, 4.0],
                [-0.71693, 1.95596, 3.52144],
                [-0.90987, 2.17318, 2.79514],
                [-1.10713, 2.39562, 2.38658],
                [-1.30725, 2.62167, 2.11603],
                [-1.50807, 2.84882, 1.91736],
                [-1.71407, 3.06774, 1.7643],
                [-1.93082, 3.26641, 1.64032],
                [-2.15977, 3.43268, 1.53467],
                [-2.39824, 3.55647, 1.53467],
                [-2.64036, 3.6311, 1.53467],
                [-2.87856, 3.65313, 1.53467],
                [-3.10433, 3.62144, 1.53467],
                [-3.30799, 3.53604, 1.53467],
                [-3.47701, 3.39788, 1.65822],
                [-3.60846, 3.22106, 1.73892],
                [-3.69878, 3.01409, 1.83066],
                [-3.74519, 2.78502, 1.93521],
                [-3.74606, 2.54219, 2.05935],
                [-3.7019, 2.29426, 2.18696],
                [-3.61578, 2.0495, 1.98047],
                [-3.49339, 1.81444, 1.98047],
                [-3.35534, 1.61048, 1.98047],
                [-3.23938, 1.40143, 1.98047],
                [-3.15121, 1.18537, 1.98047],
                [-3.09668, 0.9602, 1.98047],
                [-3.08407, 0.72346, 2.34848],
                [-3.10421, 0.47791, 2.63498],
                [-3.15394, 0.22462, 3.05529],
                [-3.22897, -0.03526, 3.6773],
                [-3.32398, -0.30049, 3.80452],
                [-3.43733, -0.58092, 2.94286],
                [-3.54544, -0.86345, 2.48315],
                [-3.64522, -1.14916, 2.18616],
                [-3.73331, -1.43894, 1.97504],
                [-3.80613, -1.73316, 1.81511],
                [-3.85984, -2.03156, 1.68502],
                [-3.88108, -2.33231, 1.57728],
                [-3.85793, -2.6263, 1.48638],
                [-3.78399, -2.89884, 1.40712],
                [-3.66094, -3.13421, 1.40712],
                [-3.49719, -3.32007, 1.40712],
                [-3.30424, -3.44861, 1.40712],
                [-3.09441, -3.51512, 1.40712],
                [-2.88048, -3.5158, 1.40712],
                [-2.6779, -3.44609, 1.57641],
                [-2.49625, -3.32139, 1.72646],
                [-2.34124, -3.15005, 1.92392],
                [-2.21599, -2.93972, 2.20724],
                [-2.12037, -2.69858, 2.02637],
                [-2.04981, -2.43574, 1.78172],
                [-1.99508, -2.16083, 1.60383],
                [-1.93263, -1.88489, 1.47009],
                [-1.84975, -1.62521, 1.36168],
                [-1.7377, -1.39302, 1.36168],
                [-1.59372, -1.19856, 1.36168],
                [-1.42019, -1.0509, 1.36168],
                [-1.22334, -0.95879, 1.36168],
                [-1.01343, -0.93121, 1.36168],
                [-0.80729, -0.97786, 1.49994],
                [-0.61965, -1.08431, 1.61331],
                [-0.45969, -1.24165, 1.75187],
                [-0.33377, -1.44167, 1.92991],
                [-0.2455, -1.67602, 1.83366],
                [-0.19492, -1.93596, 1.50247],
                [-0.1775, -2.21285, 1.3],
                [-0.18417, -2.49917, 1.3],
                [-0.19965, -2.75653, 1.3],
                [-0.18606, -2.99663, 1.3],
                [-0.12847, -3.2071, 1.3],
                [-0.02244, -3.37836, 1.3],
                [0.13308, -3.49719, 1.38569],
                [0.32489, -3.56319, 1.66601],
                [0.5401, -3.58442, 1.96655],
                [0.77185, -3.56568, 2.44337],
                [1.01401, -3.51545, 2.86799],
                [1.26577, -3.48745, 2.61789],
                [1.51687, -3.48155, 2.16026],
                [1.76715, -3.49571, 1.88052],
                [2.01654, -3.52832, 1.6859],
                [2.26497, -3.578, 1.54037],
                [2.51115, -3.64316, 1.42374],
                [2.75413, -3.67895, 1.42374],
                [2.98477, -3.6764, 1.42374],
                [3.19678, -3.63205, 1.42374],
                [3.38448, -3.54454, 1.42374],
                [3.54025, -3.41222, 1.42374],
                [3.65044, -3.23239, 1.54918],
                [3.71023, -3.01561, 1.64974],
                [3.71084, -2.77068, 1.77023],
                [3.64508, -2.51257, 1.91947],
                [3.51617, -2.26397, 2.10838],
                [3.33898, -2.04219, 2.3651],
                [3.12846, -1.85, 2.74182],
                [2.89518, -1.68254, 3.36852],
                [2.6409, -1.52898, 3.59073],
                [2.39794, -1.35761, 3.86154],
                [2.16532, -1.17111, 4.0],
                [1.94187, -0.97199, 4.0],
                [1.72626, -0.76268, 4.0],
                [1.51698, -0.54555, 4.0],
                [1.31238, -0.32295, 4.0],
                [1.11068, -0.09712, 4.0],
                [0.91754, 0.1212, 4.0],
                [0.73441, 0.32779, 4.0],
                [0.56319, 0.52056, 4.0],
                [0.40447, 0.69893, 4.0]]


def closest_two_optimal_waypoint(agent_x, agent_y): # 에이전트와 가장 가까운 최적화 waypoint 두개 반환
    distances = []
    # waypoint들 
    for waypoint in waypoints_opt:
        wp_x, wp_y = waypoint[0], waypoint[1]
        distance = math.sqrt((wp_x-agent_x)**2+(wp_y-agent_y)**2)
        distances.append([waypoint, distance])

    distances.sort(key=lambda x:x[1])
    return [distances[0][0], distances[1][0]]


def distance_from_optimal_path(agent_x, agent_y): # 가장 가까운 waypoint 두개를 이은 직선과 에이전트와의 거리 반환
    waypoint1, waypoint2 = closest_two_optimal_waypoint(agent_x, agent_y)
    wp1_x, wp1_y = waypoint1[0], waypoint1[1]
    wp2_x, wp2_y = waypoint2[0], waypoint2[1]
    
    # 벡터 계산 (wp1 -> wp2 방향 벡터)
    dx, dy = wp2_x - wp1_x, wp2_y - wp1_y
    
    # 직선 방정식 y = mx + b
    if dx == 0:  # 수직선의 경우
        return abs(agent_x - wp1_x)
    
    m = dy / dx
    b = wp1_y - m * wp1_x

    # 에이전트의 위치에서 직선까지의 수직 거리
    distance = abs(m * agent_x - agent_y + b) / math.sqrt(m**2 + 1)
    
    return distance


def get_reward_path(agent_x, agent_y): # distance_from_optimal_path로 얻은 거리가 작을수록 큰 보상 리턴 (보상범위 : 0.001 ~ 1.0)
    distance = distance_from_optimal_path(agent_x, agent_y)

    # reward = max(1e-3, 1-(distance/0.5))
    reward = max(0.1, math.exp(-distance / 0.7))
    
    return float(reward)

def get_reward_living(params):
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    reward = 1e-3

    marker1 = track_width * 0.45
    marker2 = track_width * 0.5
    if distance_from_center < marker1:
        reward = 1.0
    elif distance_from_center < marker2:
        reward = 0.5
    else:
        reward = 1e-3

    return reward


def get_reward_speed_steering(params):
    straight_zone = [[0,13],[26,35],[67,77],[89,100]]
    progress = params['progress']
    speed = params['speed']
    steering = params['steering_angle'] # 왼쪽이 양수

    if progress<=12 or 26<=progress<=35 or 67<=progress<=77 or 89<=progress: #직진구역
        if 3.0 <= speed:
            reward_speed = 1.0
        else:
            reward_speed = 0.1

        if abs(steering) < 15:
            reward_steering = 1.0
        else:
            reward_steering = 0.1
    else:
        if 1.5 <= speed <= 2.5:
            reward_speed = 1.0
        else:
            reward_speed = 0.1
        
        if 0 < abs(steering):
            reward_steering = 1.0
        else:
            reward_steering = 0.1
    
    reward = 0.6*reward_speed + 0.4*reward_steering
    return float(reward)


def reward_function(params):
    heading = params['heading']
    agent_x = params['x']
    agent_y = params['y']
    progress = params['progress']
    steps = params['steps']

    reward_speed_steering = get_reward_speed_steering(params)
    reward_living = get_reward_living(params)
    reward_path = get_reward_path(agent_x,agent_y)

    reward_progress = progress/steps/100

    reward = 1.0*reward_path + 1.0*reward_living + 0.8*reward_speed_steering + 0.0*reward_progress
    return float(reward)