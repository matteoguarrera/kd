"""
Description: Autonomous vehicle controller example
"""

import copy
import math
from utils import *



def wprint(*args):
    print("[DEBUG-WHILE]" + " ".join(map(str, args)) + " [XXX]")


# print(supervisor.getNumberOfDevices())
# # print(dir(robot))  # methods of supervisor
lst_fn = dir(lead)



set_speed(20.0)


_step_ = 0
# if has_camera:
# if enable_collision_avoidance:
while lead.step() != -1:

    _step_ += 1
    # Start the main loop

    if _step_ % (TIME_STEP / lead.getBasicTimeStep()) == 0:
        # print(_step_)
        camera_image = camera.getImage()  # this is in bytes
        # camera_image = camera.getImageArray()  # [128, 64, 3]
        sick_data = sick.getRangeImage()  # 180

        angle_tmp = process_camera_image(camera_image)
        yellow_line_angle = filter_angle(angle_tmp)

        # print(sick_data)
        if enable_collision_avoidance:
            # print(type(sick_data))
            obstacle_angle, obstacle_dist = process_sick_data(sick_data)
        else:
            obstacle_angle = UNKNOWN
            obstacle_dist = 0

        if _step_ %50 == 0:
            if angle_tmp != 'unknown':
                print(f'{angle_tmp:.4f} {yellow_line_angle:.4f}')
            else:
                print('blind')
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
