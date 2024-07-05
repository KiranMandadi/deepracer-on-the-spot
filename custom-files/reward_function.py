import math
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def control(self, error):
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class AdaptivePIDController(PIDController):
    def __init__(self, kp, ki, kd):
        super().__init__(kp, ki, kd)

    def control(self, error, speed):
        adaptive_kp = self.kp * (1 + speed / 4.0)
        adaptive_kd = self.kd * (1 + speed / 4.0)
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        return adaptive_kp * error + self.ki * self.integral + adaptive_kd * derivative

adaptive_pid = AdaptivePIDController(kp=1.0, ki=0.0, kd=0.1)

def calculate_curvature(waypoints, closest_waypoints):
    waypoint_1 = waypoints[closest_waypoints[0]]
    waypoint_2 = waypoints[closest_waypoints[1]]
    waypoint_3 = waypoints[(closest_waypoints[1] + 1) % len(waypoints)]
    curvature = np.abs(np.arctan2(waypoint_3[1] - waypoint_2[1], waypoint_3[0] - waypoint_2[0]) -
                       np.arctan2(waypoint_2[1] - waypoint_1[1], waypoint_2[0] - waypoint_1[0]))
    return curvature

def calculate_apex_distance(waypoints, closest_waypoints, x, y):
    next_waypoint = waypoints[closest_waypoints[1]]
    prev_waypoint = waypoints[closest_waypoints[0]]
    apex_point = ((next_waypoint[0] + prev_waypoint[0]) / 2, (next_waypoint[1] + prev_waypoint[1]) / 2)
    distance_from_apex = np.sqrt((x - apex_point[0])**2 + (y - apex_point[1])**2)
    return distance_from_apex

def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    speed = params['speed']
    steering_angle = abs(params['steering_angle'])
    progress = params['progress']
    steps = params['steps']
    is_offtrack = params['is_offtrack']
    all_wheels_on_track = params['all_wheels_on_track']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    waypoints = params['waypoints']
    steering_angle_change = params.get('steering_angle_change', 0.0)
    prev_steering_angles = params.get('prev_steering_angles', [steering_angle])
    prev_speed = params.get('prev_speed', speed)
    x = params['x']
    y = params['y']
    prev_x = params.get('x_prev', x)
    prev_y = params.get('y_prev', y)

    reward = 1.0

    if is_offtrack or not all_wheels_on_track:
        return 1e-3

    marker_1 = 0.1 * track_width
    marker_2 = 0.2 * track_width
    marker_3 = 0.3 * track_width

    if distance_from_center <= marker_1:
        reward += 3.0
    elif distance_from_center <= marker_2:
        reward += 2.0
    elif distance_from_center <= marker_3:
        reward += 1.0
    else:
        reward *= 0.1

    next_waypoint = waypoints[closest_waypoints[1]]
    prev_waypoint = waypoints[closest_waypoints[0]]
    track_direction = np.degrees(np.arctan2(next_waypoint[1] - prev_waypoint[1], next_waypoint[0] - prev_waypoint[0]))
    direction_diff = np.abs(track_direction - heading)

    curvature = calculate_curvature(waypoints, closest_waypoints)
    if curvature < 0.1:
        optimal_speed = 3.0
    else:
        optimal_speed = max(1.0, 3.0 - curvature * 10)

    speed_diff = abs(speed - optimal_speed)
    if speed_diff < 0.1:
        reward += 2.0
    elif speed_diff < 0.2:
        reward += 1.0
    else:
        reward += 0.5

    if speed > optimal_speed and curvature > 0.1:
        reward *= 0.4

    SPEED_STABILITY_THRESHOLD = 0.1
    if np.abs(speed - prev_speed) < SPEED_STABILITY_THRESHOLD:
        reward += 1.5

    BRAKING_THRESHOLD = 0.3
    if prev_speed - speed > BRAKING_THRESHOLD:
        reward *= 0.8

    ABS_STEERING_THRESHOLD = 0.3
    if steering_angle > ABS_STEERING_THRESHOLD:
        reward *= 0.3

    OSCILLATION_THRESHOLD = 0.2
    if len(prev_steering_angles) > 1:
        steering_deltas = np.abs(np.diff(prev_steering_angles[-5:]))
        if np.any(steering_deltas > OSCILLATION_THRESHOLD):
            reward *= 0.4

    SMOOTH_STEERING_THRESHOLD = 0.15
    if steering_angle_change < SMOOTH_STEERING_THRESHOLD:
        reward += 2.0

    DIRECTION_THRESHOLD = 2.0
    if direction_diff > DIRECTION_THRESHOLD:
        reward *= 0.3

    apex_distance = calculate_apex_distance(waypoints, closest_waypoints, x, y)
    if apex_distance < 0.1 * track_width:
        reward += 2.0
    elif apex_distance < 0.2 * track_width:
        reward += 1.0
    else:
        reward *= 0.5

    if curvature < 0.1 and speed > optimal_speed:
        reward += 1.0

    if steering_angle_change > 0.3:
        reward *= 0.8

    steering_error = steering_angle_change
    pid_correction = adaptive_pid.control(steering_error, speed)
    if abs(pid_correction) < 0.2:
        reward += 2.0

    if steps > 1 and speed > prev_speed:
        reward += 1.0

    prev_x, prev_y = waypoints[closest_waypoints[0]]
    distance_traveled = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
    OPTIMAL_DISTANCE_PER_STEP = 0.4
    if distance_traveled <= OPTIMAL_DISTANCE_PER_STEP:
        reward += 1.0
    else:
        reward *= 0.8

    reward += (progress / 100.0) * 1.5

    TOTAL_NUM_STEPS = 300
    if progress == 100:
        reward += 100 * (1 - (steps / TOTAL_NUM_STEPS))
    elif progress > 0:
        reward += progress / TOTAL_NUM_STEPS * 10

    SPEED_CONSISTENCY_THRESHOLD = 0.1
    if np.abs(speed - prev_speed) < SPEED_CONSISTENCY_THRESHOLD:
        reward += 1.2

    reward += (progress / 100.0) * 2.0

    MILESTONE_REWARD = 5.0
    if progress >= 25 and steps < 75:
        reward += MILESTONE_REWARD
    if progress >= 50 and steps < 150:
        reward += MILESTONE_REWARD
    if progress >= 75 and steps < 225:
        reward += MILESTONE_REWARD

    distance_traveled = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
    reward += distance_traveled * 0.1

    if steering_angle > 15:
        reward *= 0.8
    elif steering_angle > 10:
        reward *= 0.9
    else:
        reward += 0.5

    return float(reward)
