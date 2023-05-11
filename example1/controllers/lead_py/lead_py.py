"""
Description: Autonomous vehicle controller example
"""

import sys
import os
# print(os.listdir('../../../segment'))
sys.path.append('../../..')
from segment.model import UNET

from utils import *
import torch


def wprint(*args):
    print("[DEBUG-WHILE]" + " ".join(map(str, args)) + " [XXX]")


# print(supervisor.getNumberOfDevices())
# # print(dir(robot))  # methods of supervisor
lst_fn = dir(lead)


def load_nn():
    teacher = UNET(layers=[3, 64, 128], classes=10).to('cpu')  # [3, 64, 128] # 256, 512, 1024
    TEACHER_PATH = '../../../segment/v0.97_teacher_l3_64_128_e10_lr5e-05_d05_07_23_04_02'
    checkpoint = torch.load(TEACHER_PATH, map_location=torch.device('cpu'))
    teacher.load_state_dict(checkpoint['model_state_dict'])
    return teacher


def process_camera_image_nn(image_array):
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
    pixelwise_corr = np.sum(pred_line)  # count yellow (?)

    x_pos, y_pos = np.where(pred_line)
    # x_pos, y_pos,
    # if no pixels was detected...
    if pixelwise_corr <= 10:
        return 'unknown', None, None

    angle_nn = (sum(x_pos) / pixelwise_corr / camera_width - 0.5) * camera_fov
    return angle_nn, sumx, pixel_count


model = load_nn()

set_speed(40.0)

_img_saved_ = 0
_step_ = 0
# if has_camera:
# if enable_collision_avoidance:
while lead.step() != -1:

    _step_ += 1
    # Start the main loop

    if _step_ % (TIME_STEP / lead.getBasicTimeStep()) == 0:
        # print(_step_)
        camera_image = camera.getImage()  # this is in bytes
        camera_image_array = camera.getImageArray()  # [128, 64, 3]
        sick_data = sick.getRangeImage()  # 180

        angle_tmp, sumx, pixel_count = process_camera_image(camera_image)
        angle_nn, sumx_nn, pixel_count_nn = process_camera_image(camera_image)

        yellow_line_angle = filter_angle(angle_nn)
        # yellow_line_angle = filter_angle(angle_tmp)

        # print(sick_data)
        if enable_collision_avoidance:
            # print(type(sick_data))
            obstacle_angle, obstacle_dist = process_sick_data(sick_data)
        else:
            obstacle_angle = UNKNOWN
            obstacle_dist = 0

        # if _step_ %50 == 0:
        #     if angle_tmp != 'unknown':
        #         print(f'{angle_tmp:.4f} {yellow_line_angle:.4f}')
        #     else:
        #         print('blind')
        avoid_obstacles_and_follow_yellow_line(obstacle_dist, obstacle_angle, yellow_line_angle)

        compute_gps_speed()

#
# from utils import *
# # from controller import Camera, Device, Display, GPS, Keyboard, Lidar, Robot, VehicleDriver
# import math
# import time
#
# # To be used as array indices
# X, Y, Z = range(3)
#
# # Initialize the Robot
# robot = Robot()
#
# # Initialize the devices
# if robot.getDevice("gps"):
#     gps = GPS("gps")
#     has_gps = True
#
# if robot.getDevice("camera"):
#     camera = Camera("camera")
#     has_camera = True
#     camera.enable(TIME_STEP)
#
# if robot.getDevice("display"):
#     display = Display("display")
#     display_width = display.getWidth()
#     display_height = display.getHeight()
#     speedometer_image = display.imageLoad("speedometer.png")
#
# if robot.getDevice("sick_lidar"):
#     sick = Lidar("sick_lidar")
#     sick_width = sick.getHorizontalResolution()
#     sick_range = sick.getMaxRange()
#     sick_fov = sick.getHorizontalFov()
#     sick.enable(TIME_STEP)
#
# driver = VehicleDriver(robot, TIME_STEP)
# driver.setSteeringAngle(0.0)
# driver.setCruisingSpeed(20.0)
#
# # Start the main loop
# while robot.step(TIME_STEP) != -1:
#     if has_camera:
#         # Get camera data
#         camera_data = camera.getImage()
#         if camera_data is not None:
#             camera_width = camera.getWidth()
#             camera_height = camera.getHeight()
#
#     if has_gps:
#         # Get GPS data
#         gps_data = gps.getValues()
#         gps_coords[X] = gps_data[X]
#         gps_coords[Y] = gps_data[Y]
#         gps_coords[Z] = gps_data[Z]
#         gps_speed = gps.getSpeed()
#
#     if sick
