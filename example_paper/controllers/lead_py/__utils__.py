
#
# # Enable various 'features'
# enable_collision_avoidance = False
# enable_display = False
# has_gps = False
# has_camera = False
#
# # Camera
# camera = None
# camera_width = -1
# camera_height = -1
# camera_fov = -1.0
#
# # SICK laser
# sick = None
# sick_width = -1
# sick_range = -1.0
# sick_fov = -1.0
#
# # Speedometer
# display = None
# display_width = 0
# display_height = 0
# speedometer_image = None
#

#
# r = -1
#
#
# # print help message
# def print_help():
#     print("You can drive this car!")
#     print("Select the 3D window and then use the cursor keys to:")
#     print("[LEFT]/[RIGHT] - steer")
#     print("[UP]/[DOWN] - accelerate/slow down")
#
#
# # set autodrive
# def set_autodrive(onoff):
#     global autodrive, has_camera
#     if autodrive == onoff:
#         return
#     autodrive = onoff
#     if not autodrive:
#         print("switching to manual drive...")
#         print("hit [A] to return to auto-drive.")
#     else:
#         if has_camera:
#             print("switching to auto-drive...")
#         else:
#             print("impossible to switch auto-drive on without camera...")
#
#
# # set target speed
# def set_speed(kmh):
#     global speed
#     # max speed
#     if kmh > 250.0:
#         kmh = 250.0
#     speed = kmh
#     # print("setting speed to %g km/h" % kmh)
#     lead.setVelocity(kmh)
#
#

#
# # change manual steering angle
# def change_manual_steer_angle(inc):
#     global manual_steering
#     set_autodrive(False)
#     new_manual_steering = manual_steering + inc
#     if new_manual_steering <= 25.0 and new_manual_steering >= -25.0:
#         manual_steering = new_manual_steering
#         set_steering_angle(manual_steering * 0.02)
#     if manual_steering == 0:
#         print("going straight")
#     else:
#         print("turning %.2f rad (%s)" % (steering_angle, "left" if steering_angle < 0 else "right"))
#
#
# # Unused so not converted
# # def void check_keyboard()
#
# def color_diff(a, b):
#     diff = 0
#     for i in range(3):
#         d = a[i] - b[i]
#         diff += d if d > 0 else -d
#     return diff
#
#
# # // returns approximate angle of yellow road line
# # // or UNKNOWN if no pixel of yellow line visible

#
# # filter angle of the yellow line (simple average)
# def filter_angle(new_value):
#     global first_call
#     global old_value
#     old_value = [None]*FILTER_SIZE
#     first_call = True
#
#     if first_call or new_value == UNKNOWN:  # reset all the old values to 0.0
#         first_call = False
#         old_value = [0]*FILTER_SIZE
#     else:  # shift old values
#         old_value[:-1] = old_value[1:]
#
#     if new_value == UNKNOWN:
#         return UNKNOWN
#     else:
#         old_value[-1] = new_value
#         return sum(old_value) / FILTER_SIZE
#
#

#
# # class GPS (Device):
# #     def enable(self, samplingPeriod):
# #     def disable(self):
# #     def getSamplingPeriod(self):
# #     def getValues(self):
# #     def getSpeed(self):
# #     def getSpeedVector(self):
# def compute_gps_speed():
#     coords = gps.getValues()
#     speed_ms = gps.getSpeed()
#     # store into global variables
#     global gps_speed, gps_coords
#     gps_speed = speed_ms * 3.6  # convert from m/s to km/h
#     gps_coords = copy.deepcopy(coords)
#
# def applyPID(yellow_line_angle):
#     global PID_need_reset
#     global oldValue, integral
#
#     if PID_need_reset:
#         oldValue = yellow_line_angle
#         integral = 0.0
#         PID_need_reset = False
#
#     # anti-windup mechanism
#     if math.copysign(1, yellow_line_angle) != math.copysign(1, oldValue):
#         integral = 0.0
#
#     diff = yellow_line_angle - oldValue
#
#     # limit integral
#     if -30 <= integral <= 30:
#         integral += yellow_line_angle
#
#     oldValue = yellow_line_angle
#     return KP * yellow_line_angle + KI * integral + KD * diff
#
# def main():
#     lead = Driver()
#     robot = Robot() #what is the difference?
#
#     camera1 = lead.getDevice("camera")
#     camera2 = =robot.getDevice("camera")
#     raise NotImplementedError
#
#     # check how many device
#     # class Robot:
#     #     def getNumberOfDevices(self):
#     #     def getDeviceByIndex(self, index):
#     # class Driver:
#     #     def setCruisingSpeed(self, speed):
#     #     def getTargetCruisingSpeed(self):
#     #     def setSteeringAngle(self, steeringAngle):
#     #     def getSteeringAngle(self):
#
#     robot_lead = lead  # robot
#     num_device = robot_lead.getNumberOfDevices()
#     for i in range(num_device):
#         dev_obj = robot_lead.getDeviceByIndex(i)
#         dev_name =    dev_obj.getName()
#         if dev_name == "Sick LMS 291":
#             enable_collision_avoidance = True
#         elif dev_name == "display":
#             enable_display = True
#         elif dev_name == "gps":
#             has_gps = True
#         elif dev_name == "camera":
#             has_camera = True
#
#     if has_camera:
#         print("name: camera")
#         camera = robot_lead
#
#         # @mat Should be good
#         camera.enable(TIME_STEP)
#         camera_width = camera.getWidth()
#         camera_height = camera.getHeight()
#         camera_fov = camera.getFov()
#         raise NotImplementedError
#
#
#     if enable_collision_avoidance:
#         print("name: Sick LMS 291")
#         sick = robot_lead
#
#         # @mat Should be good
#         sick.enable(TIME_STEP);
#         sick_width = sick.getHorizontalResolution()
#         sick_range = sick.getMaxRange()
#         sick_fov = sick.getFov()
#         raise NotImplementedError
#
#     if has_gps:
#         print("name: gps")
#         gps = robot_lead
#
#         gps.enable(TIME_STEP)
#         raise NotImplementedError
#
#
#     if enable_display:
#         print("name: display")
#         print("Don't print for now")
#         raise  NotImplementedError
#
#
#         # self.distanceSensor = self.getDistanceSensor('my_distance_sensor')
#         # self.led = self.getLed('my_led')
#         #
#
#     # getDistanceSensor()
#     sensor = robot.getDevice("my_distance_sensor")
#     sensor.enable(TIME_STEP)
#
#     set_speed(20.0) # km/h
#
#
#     # driver class
#     robot_lead.setHazardFlashers(True)
#     robot_lead.setDippedBeams(True)
#     robot_lead.getAntifogLights(True)
#     robot_lead.setWiperMode(SLOW)
#
#     # Start the main loop
#     while robot.step(TIME_STEP) != -1:
#         print("[LEAD]")
#
#         if has_camera:
#             camera_image = camera.getImage()
#         if enable_collision_avoidance:
#             sick_data = sick.getRangeImage()
#
#         if autodrive and has_camera:
#             angle_tmp = process_camera_image(camera_image)
#             yellow_line_angle = filter_angle(angle_tmp)
#             if enable_collision_avoidance:
#                 obstacle_angle = process_sick_data(sick_data, obstacle_dist)
#             else:
#               obstacle_angle = UNKNOWN
#               obstacle_dist = 0
#
#             if enable_collision_avoidance and obstacle_angle != UNKNOWN:
#                 # an obstacle has been detected
#                 # compute the steering angle required to avoid the obstacle
#                 obstacle_steering = steering_angle
#                 if 0.0 < obstacle_angle < 0.4:
#                     obstacle_steering = steering_angle + (obstacle_angle - 0.25) / obstacle_dist
#                 elif obstacle_angle > -0.4:
#                     obstacle_steering = steering_angle + (obstacle_angle + 0.25) / obstacle_dist
#
#                 steer = steering_angle
#                 # if we see the line we determine the best steering angle to both avoid obstacle and follow the line
#                 if yellow_line_angle != UNKNOWN:
#                     line_following_steering = applyPID(yellow_line_angle)
#                     if obstacle_steering > 0 and line_following_steering > 0:
#                         steer = obstacle_steering if obstacle_steering > line_following_steering else line_following_steering
#                     elif obstacle_steering < 0 and line_following_steering < 0:
#                         steer = obstacle_steering if obstacle_steering < line_following_steering else line_following_steering
#                 else:
#                     PID_need_reset = True
#                 # apply the computed required angle
#                 set_steering_angle(steer)
#
#
#
# # Check that!
#
# # if yellow_line_angle != UNKNOWN:
# #     # no obstacle has been detected, simply follow the line
# #     if r < 0:
# #         wbu_driver_set_brake_intensity(0.0)
# #     else:
# #         wbu_driver_set_brake_intensity(0.7)
# #         r = -1
# #         PID_need_reset = True
# #     set_steering_angle(applyPID(yellow_line_angle))
# # else:
# #     # no obstacle has been detected but we lost the line => we brake and hope to find the line again
# #     wbu_driver_set_brake_intensity(0.7)
# #     if r < 0:
# #         if argv[1] == "L":
# #             r = 0
# #         elif argv[1] == "R":
# #             r = 1
# #         elif argv[1] == "S":
# #             r = 2
# #     if r == 0:
# #         set_steering_angle(applyPID(-0.5))
# #     elif r == 1:
# #         set_steering_angle(applyPID(0.5))
# #     else:
# #         wbu_driver_set_brake_intensity(1.0)
# #         if r > 10:
# #             set_steering_angle(applyPID(0.0))
# #         else:
# #             r += 1
# #         # set_steering_angle(applyPID(steering_angle))
# #         # steering_angle *= 0.9
