# matteogu@berkeley.edu

# Bunch of initialization and constants
try:
    from controller import Supervisor
    from controller import GPS
    # from controller import Device
    from controller import Camera
    from controller import Robot
    from vehicle import Driver

except ModuleNotFoundError:
    import sys

    sys.exit("This functionality requires webots to be installed")

import sys
sys.path.append('../../..')
from segment.model import *


def dprint(*args):
    pass
    # print("[DEBUG-DEVICE]" + " ".join(map(str, args)) + " [XXX]")


TIME_STEP = 50
UNKNOWN = 'unknown'

# Line following PID
KP = 0.25
KI = 0.006
KD = 2

PID_need_reset = False
oldValue = 0.0
integral = 0.0

# Size of the yellow line angle filter
FILTER_SIZE = 3

# GPS
gps_coords = [0.0, 0.0, 0.0]
gps_speed = 0.0

# Misc variables
speed = 0.0
steering_angle = 0.0
manual_steering = 0
autodrive = True

r = -1

# enable various 'features'
enable_collision_avoidance = True
enable_display = True
has_gps = True
has_camera = True


lead = Driver()  # leader car

dprint('number device: ', lead.getNumberOfDevices())
camera = lead.getDevice(name='camera')
camera.enable(TIME_STEP)
camera_width = camera.getWidth()
camera_height = camera.getHeight()
camera_fov = camera.getFov()
dprint(camera, camera_width, camera_height, camera_fov)

sick = lead.getDevice(name='Sick LMS 291')
sick.enable(TIME_STEP)
sick_width = sick.getHorizontalResolution()
sick_range = sick.getMaxRange()
sick_fov = sick.getFov()
dprint(sick, sick_width, sick_range, sick_fov)


gps = lead.getDevice(name='gps')
gps.enable(TIME_STEP)
dprint(gps)

display = lead.getDevice(name='display')
dprint(gps, camera, display, sick)


# driver class
lead.setHazardFlashers(True)
lead.setDippedBeams(True)
lead.setAntifogLights(True)
lead.setWiperMode(1)  # DOWN 0, SLOW 1, NORMAL 2, FAST 3

# filter_angle helper
first_call = True
old_value = [None]*FILTER_SIZE


# compute rgb difference
def __color_diff__(a, b):
    diff = 0
    for i in range(3):
        d = a[i] - b[i]
        diff += d if d > 0 else -d
    return diff


def load_nn(arch='teacher'):
    if arch == 'student':
        model = UNET(layers=[3, 64], classes=10).to('cpu')  # [3, 64, 128] # 256, 512, 1024
        PATH = '../../../segment/v0.91_student_l3_64_e20_lr5e-05_d05_07_23_38_10'
    elif arch == 'teacher':
        model = UNET(layers=[3, 64, 128], classes=10).to('cpu')  # [3, 64, 128] # 256, 512, 1024
        PATH = '../../../segment/v0.97_teacher_l3_64_128_e10_lr5e-05_d05_07_23_04_02'
    else:
        raise NotImplementedError

    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def process_camera_image_nn(model, image_array):
    # image processing to adapt to how the model was trained
    img = torch.tensor(image_array)
    img = torch.reshape(img, shape=(64, 128, 3))
    img = img.permute(2, 1, 0)
    img = img.unsqueeze(0).float()  # input is float 0. 255.

    pred = model(img)  #
    preds_class = torch.argmax(pred, dim=1)  # input [1, 10, 128, 64], output [1, 128, 64]) torch.int64
    pred_line = (9 == preds_class.int()).numpy()[0]  # remove redundant dimension
    # print(pred_line)
    # need to compare the output with the pixel
    pixel_wise_corr = np.sum(pred_line)  # count yellow (?)

    x_pos, y_pos = np.where(pred_line)
    # x_pos, y_pos,
    # if no pixels was detected...
    if pixel_wise_corr <= 10:
        return 'unknown', None, None

    angle_nn = (sum(x_pos) / pixel_wise_corr / camera_width - 0.5) * camera_fov
    return angle_nn, sum(x_pos), pixel_wise_corr


# returns approximate angle of yellow road line
# or UNKNOWN if no pixel of yellow line visible
def process_camera_image(image):
    global camera_height, camera_width, camera_fov
    num_pixels = camera_height * camera_width  # number of pixels in the image
    REF = [95, 187, 203]  # road yellow (BGR format)
    sumx = 0  # summed x position of pixels
    pixel_count = 0  # yellow pixels count
    for x in range(num_pixels):
        pixel = image[4 * x:4 * x + 3]
        if __color_diff__(pixel, REF) < 30:
            sumx += x % camera_width
            pixel_count += 1  # count yellow pixels

    # if no pixels was detected...
    if pixel_count == 0:
        return 'unknown', None, None

    return (sumx / pixel_count / camera_width - 0.5) * camera_fov, sumx, pixel_count


# filter angle of the yellow line (simple average)
def filter_angle(new_value):
    global first_call, old_value, FILTER_SIZE
    if first_call or new_value == 'unknown':  # reset all the old values to 0.0
        first_call = False
        old_value = [0.0] * FILTER_SIZE
    else:  # shift old values
        old_value[:-1] = old_value[1:]

    if new_value == 'unknown':
        return 'unknown'
    else:
        old_value[-1] = new_value
        _sum_ = 0.0
        for i in range(FILTER_SIZE):
            _sum_ += old_value[i]
        return _sum_ / FILTER_SIZE


# returns approximate angle of obstacle
# or UNKNOWN if no obstacle was detected
def process_sick_data(sick_data_local):
    HALF_AREA = 20  # check 20 degrees wide middle area
    sumx = 0
    collision_count = 0
    obstacle_dist = 0.0
    for x in range(sick_width // 2 - HALF_AREA, sick_width // 2 + HALF_AREA):
        range_value = sick_data_local[x]
        # print(range_value)
        if range_value < 20.0:
            sumx += x
            collision_count += 1
            obstacle_dist += range_value

    # if no obstacle was detected...
    if collision_count == 0:
        return 'unknown', obstacle_dist

    obstacle_dist /= collision_count
    return (sumx / collision_count / sick_width - 0.5) * sick_fov, obstacle_dist


def applyPID(yellow_line_angle):
    global PID_need_reset, oldValue, integral

    # oldValue = 0.0  # not sure this is right
    # integral = 0.0

    if PID_need_reset:
        oldValue = yellow_line_angle
        integral = 0.0
        PID_need_reset = False

    # anti-windup mechanism
    if (yellow_line_angle < 0) != (oldValue < 0):
        integral = 0.0

    diff = yellow_line_angle - oldValue

    # limit integral
    if -30 < integral < 30:
        integral += yellow_line_angle

    oldValue = yellow_line_angle
    res = KP * yellow_line_angle + KI * integral + KD * diff
    # file = open('py_res.csv', 'a')
    # file.write(f'{res}, ')
    # file.close()

    return res


# set target speed
def set_speed(kmh):
    global speed
    # max speed
    if kmh > 250.0:
        kmh = 250.0

    speed = kmh
    # printf("setting speed to %g km/h\n", kmh);
    lead.setCruisingSpeed(kmh)


# set steering angle
def set_steering_angle(wheel_angle):
    global steering_angle

    # limit the difference with previous steering_angle
    if wheel_angle - steering_angle > 0.1:
        wheel_angle = steering_angle + 0.1

    if wheel_angle - steering_angle < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle

    # limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5

    lead.setSteeringAngle(wheel_angle)
    # lead.setVelocity((speed / 3.6) - (wheel_angle * 0.5))
    # motor_right.setVelocity((speed / 3.6) + (wheel_angle * 0.5))


def avoid_obstacles_and_follow_yellow_line(obstacle_dist, obstacle_angle, yellow_line_angle):
    global PID_need_reset, r, steering_angle, enable_collision_avoidance

    # avoid obstacles and follow yellow line
    if enable_collision_avoidance and (obstacle_angle != 'unknown'):

        # an obstacle has been detected
        # compute the steering angle required to avoid the obstacle
        obstacle_steering = steering_angle
        if 0.0 < obstacle_angle < 0.4:
            obstacle_steering = steering_angle + (obstacle_angle - 0.25) / obstacle_dist
        elif obstacle_angle > -0.4:
            obstacle_steering = steering_angle + (obstacle_angle + 0.25) / obstacle_dist
        steer = steering_angle
        # if we see the line we determine the best steering angle to both avoid obstacle and follow the line
        if yellow_line_angle != 'unknown':
            line_following_steering = applyPID(yellow_line_angle)
            if obstacle_steering > 0 and line_following_steering > 0:
                steer = max(obstacle_steering, line_following_steering)
            elif obstacle_steering < 0 and line_following_steering < 0:
                steer = min(obstacle_steering, line_following_steering)
        else:
            PID_need_reset = True
        # apply the computed required angle
        set_steering_angle(steer)

    elif yellow_line_angle != 'unknown':
        # no obstacle has been detected, simply follow the line
        if r < 0:
            lead.setBrakeIntensity(0.0)
        else:
            lead.setBrakeIntensity(0.7)
            r = -1
            PID_need_reset = True
        set_steering_angle(applyPID(yellow_line_angle))

    else:
        # no obstacle has been detected but we lost the line => we brake and hope to find the line again
        lead.setBrakeIntensity(0.7)
        # -- -- -- -- New stragegy, just keep straight
        set_steering_angle(applyPID(0.0))


        # if r < 0:
        #     # r = random.randint(0, 2)
        #     # if argv[1] == "L":
        #     #     r = 0
        #     # elif argv[1] == "R":
        #     #     r = 1
        #     # elif argv[1] == "S":
        #     #     r = 2
        #     r = r
        #
        # if r == 0:
        #     set_steering_angle(applyPID(-0.5))
        # elif r == 1:
        #     set_steering_angle(applyPID(0.5))
        # else:
        #     lead.setBrakeIntensity(1.0)
        #     if r > 10:
        #         set_steering_angle(applyPID(0.0))
        #     else:
        #         r += 1
        #     # set_steering_angle(applyPID(steering_angle))
        #     # steering_angle *= 0.9
        # # print(r)



def compute_gps_speed():
    global gps_speed
    coords = gps.getValues()
    speed_ms = gps.getSpeed()
    # store into global variables
    gps_speed = speed_ms * 3.6;  # convert from m/s to km/h
