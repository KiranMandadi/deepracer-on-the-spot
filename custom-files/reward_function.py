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

def calculate_look_ahead_distance(waypoints, closest_waypoints, x, y):
    look_ahead_index = (closest_waypoints[1] + 5) % len(waypoints)
    look_ahead_point = waypoints[look_ahead_index]
    distance_from_look_ahead = np.sqrt((x - look_ahead_point[0])**2 + (y - look_ahead_point[1])**2)
    return distance_from_look_ahead

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

    # Strong penalty for going off-track
    if is_offtrack or not all_wheels_on_track:
        return 1e-3

    # Reward for following the optimal path (using the apex distance)
    optimal_path_deviation = calculate_apex_distance(waypoints, closest_waypoints, x, y)
    if optimal_path_deviation < 0.1 * track_width:
        reward += 3.0
    elif optimal_path_deviation < 0.2 * track_width:
        reward += 1.5
    else:
        reward *= 0.5

    # Reward for staying close to the center line (with lower weight than optimal path)
    marker_1 = 0.1 * track_width
    marker_2 = 0.2 * track_width
    if distance_from_center <= marker_1:
        reward += 1.0
    elif distance_from_center <= marker_2:
        reward += 0.5
    else:
        reward *= 0.1

    # Calculate track direction and direction difference
    next_waypoint = waypoints[closest_waypoints[1]]
    prev_waypoint = waypoints[closest_waypoints[0]]
    track_direction = np.degrees(np.arctan2(next_waypoint[1] - prev_waypoint[1], next_waypoint[0] - prev_waypoint[0]))
    direction_diff = np.abs(track_direction - heading)

    # Penalize for large direction differences
    DIRECTION_THRESHOLD = 10.0
    if direction_diff > DIRECTION_THRESHOLD:
        reward *= 0.5

    # Calculate curvature and set optimal speed
    curvature = calculate_curvature(waypoints, closest_waypoints)
    if curvature < 0.1:
        optimal_speed = 3.5
    else:
        optimal_speed = max(1.0, 3.5 - curvature * 10)  # Reduced optimal speed for curves

    # Reward for maintaining optimal speed
    speed_diff = abs(speed - optimal_speed)
    if speed_diff < 0.1:
        reward += 2.0
    elif speed_diff < 0.2:
        reward += 1.0
    else:
        reward += 0.5

    # Stronger penalty for going too fast in curves
    if speed > optimal_speed and curvature > 0.1:
        reward *= 0.2

    # Reward for smooth speed transitions
    SPEED_STABILITY_THRESHOLD = 0.1
    if np.abs(speed - prev_speed) < SPEED_STABILITY_THRESHOLD:
        reward += 1.5

    # Penalize for excessive braking
    BRAKING_THRESHOLD = 0.3
    if prev_speed - speed > BRAKING_THRESHOLD:
        reward *= 0.8

    # Penalize for too much steering (to prevent zig-zag behavior)
    ABS_STEERING_THRESHOLD = 0.15  # More strict threshold for steering
    if steering_angle > ABS_STEERING_THRESHOLD:
        reward *= 0.1

    # Penalize for frequent steering changes
    OSCILLATION_THRESHOLD = 0.1  # Stricter threshold for frequent changes
    if len(prev_steering_angles) > 1:
        steering_deltas = np.abs(np.diff(prev_steering_angles[-5:]))
        if np.any(steering_deltas > OSCILLATION_THRESHOLD):
            reward *= 0.4

    # Penalize absolute steering changes within a short time window
    SHORT_TIME_WINDOW = 0.2  # 0.2 seconds
    if steering_angle_change > SHORT_TIME_WINDOW:
        reward *= 0.5

    # Reward for smooth steering
    SMOOTH_STEERING_THRESHOLD = 0.05  # Stricter threshold for smooth steering
    if steering_angle_change < SMOOTH_STEERING_THRESHOLD:
        reward += 2.0

    # Encourage gradual steering changes
    STEERING_GRADUAL_THRESHOLD = 0.1
    if steering_angle_change < STEERING_GRADUAL_THRESHOLD:
        reward += 1.0

    # Reward for consistent steering behavior over multiple steps
    CONSISTENT_STEERING_THRESHOLD = 0.05
    if len(prev_steering_angles) > 2:
        recent_steering_changes = np.abs(np.diff(prev_steering_angles[-3:]))
        if np.all(recent_steering_changes < CONSISTENT_STEERING_THRESHOLD):
            reward += 1.5

    # Reward for maintaining an optimal look-ahead distance
    look_ahead_distance = calculate_look_ahead_distance(waypoints, closest_waypoints, x, y)
    OPTIMAL_LOOK_AHEAD_DISTANCE = 5.0
    if look_ahead_distance < OPTIMAL_LOOK_AHEAD_DISTANCE:
        reward += 2.0
    else:
        reward *= 0.8

    # Reward for maximizing speed on straight sections
    if curvature < 0.1 and speed > 3.0:  # Ensure high speed on straight paths
        reward += 3.0  # Increased reward for high speed on straight paths

    # Penalize for unnecessary steering adjustments
    if steering_angle_change > 0.3:
        reward *= 0.8

    # Use PID controller for steering correction
    steering_error = steering_angle_change
    pid_correction = adaptive_pid.control(steering_error, speed)
    if abs(pid_correction) < 0.1:  # Stricter threshold for PID correction
        reward += 2.0

    # Reward for higher exit speed rather than entry speed
    if steps > 1 and speed > prev_speed:
        reward += 1.0

    # Penalize for longer paths (encourage taking the shortest path)
    prev_x, prev_y = waypoints[closest_waypoints[0]]
    distance_traveled = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
    OPTIMAL_DISTANCE_PER_STEP = 0.4
    if distance_traveled <= OPTIMAL_DISTANCE_PER_STEP:
        reward += 1.0
    else:
        reward *= 0.8

    # Progress-based reward
    reward += (progress / 100.0) * 1.5

    # Additional reward for completing the track faster
    TOTAL_NUM_STEPS = 300
    if progress == 100:
        reward += 100 * (1 - (steps / TOTAL_NUM_STEPS))
    elif progress > 0:
        reward += progress / TOTAL_NUM_STEPS * 10

    # Reward for consistency in speed
    SPEED_CONSISTENCY_THRESHOLD = 0.1
    if np.abs(speed - prev_speed) < SPEED_CONSISTENCY_THRESHOLD:
        reward += 1.2

    # Incremental Progress Reward
    reward += (progress / 100.0) * 2.0

    # Time-based milestones
    MILESTONE_REWARD = 5.0
    if progress >= 25 and steps < 75:
        reward += MILESTONE_REWARD
    if progress >= 50 and steps < 150:
        reward += MILESTONE_REWARD
    if progress >= 75 and steps < 225:
        reward += MILESTONE_REWARD

    # Reward for covering more distance in fewer steps
    distance_traveled = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
    reward += distance_traveled * 0.1

    # Penalize high steering angles
    if steering_angle > 15:
        reward *= 0.8
    elif steering_angle > 10:
        reward *= 0.9
    else:
        reward += 0.5

    # Advanced complex rewards

    # Penalize for heading deviation
    HEADING_THRESHOLD = 10.0
    if direction_diff > HEADING_THRESHOLD:
        reward *= 0.7

    # Reward for consistent progress without sudden stops
    if speed > 1.0:
        reward += 1.0

    # Reward for maintaining higher speeds in straight sections
    STRAIGHT_SPEED_REWARD = 1.0
    if curvature < 0.1 and speed > 2.5:
        reward += STRAIGHT_SPEED_REWARD

    # Penalize for excessive steering angle change over time
    STEERING_CHANGE_THRESHOLD = 5.0
    if len(prev_steering_angles) > 1:
        steering_changes = [abs(prev_steering_angles[i] - prev_steering_angles[i-1]) for i in range(1, len(prev_steering_angles))]
        avg_steering_change = sum(steering_changes) / len(steering_changes)
        if avg_steering_change > STEERING_CHANGE_THRESHOLD:
            reward *= 0.8

    return float(reward)
