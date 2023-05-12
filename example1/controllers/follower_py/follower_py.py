"""
Description: Autonomous vehicle controller example
"""


from utils_follower import *
import torch


def wprint(*args):
    print("[DEBUG-WHILE]" + " ".join(map(str, args)) + " [XXX]")


# print(supervisor.getNumberOfDevices())
# # print(dir(robot))  # methods of supervisor
lst_fn = dir(lead)

arch = 'student'
# arch = 'teacher'
model = load_nn(arch=arch)
print(arch)
set_speed(40.0)


np_data = [[None], [None], [None]]

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

        # angle_tmp, sumx, pixel_count = process_camera_image(camera_image)
        angle_nn, sumx_nn, pixel_count_nn = process_camera_image_nn(model, camera_image_array)

        yellow_line_angle = filter_angle(angle_nn)
        # yellow_line_angle = filter_angle(angle_tmp)

        # print(sick_data)
        if enable_collision_avoidance:
            # print(type(sick_data))
            obstacle_angle, obstacle_dist = process_sick_data(sick_data)
        else:
            obstacle_angle = UNKNOWN
            obstacle_dist = 0

        if _step_ % 1 == 0:
            # if angle_tmp != 'unknown':
            if yellow_line_angle != 'unknown':
                print(f'YL {yellow_line_angle:.4f}') # {angle_tmp:.4f}
            if angle_nn != 'unknown':
                print(f'NN {angle_nn:.4f}')
            if angle_nn == 'unknown' or yellow_line_angle == 'unknown':
                print(angle_nn, yellow_line_angle)

        avoid_obstacles_and_follow_yellow_line(obstacle_dist, obstacle_angle, yellow_line_angle)

        gps_coords = gps.getValues()  # x, y, z
        gps_speed = gps.getSpeed()
        # print(gps_coords[0], gps_coords[1], gps_speed)
        np_data[0].append(gps_coords[0])
        np_data[1].append(gps_coords[1])
        np_data[2].append(gps_speed)

        if _step_ > 10000:
            with open(f'../coords_{arch}.npy', 'wb') as f:
                np.save(f, np.array(np_data))
            print('arch ended')
            raise NotImplementedError

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
