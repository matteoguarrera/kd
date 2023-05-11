# TODOs:
#   1) Add the orientation specification. (Postpone)
#   2) Should we terminate as soon as we falsify? Both.
#   3) Run monolithic falsification.
#   4) Write seperate Scenic programs for each subscenarios.
#   5) Write a script that runs these Scenic programs by using VerifAI for each subscenario for falsifying, i.e., write a script implementing the falsification algorithm of Scenic-SMC.

# Problem: how can we have realistic physics for each subscenario?

import subprocess
import time

cmd_str = "cd ~/Documents/kd/example1/controllers/follower/build &&  make"  # cmake .. &&
subprocess.run(cmd_str, shell=True)
cmd_str = "cd ~/Documents/kd/example1/controllers/lead/build && make"  # cmake .. &&
subprocess.run(cmd_str, shell=True)

try:
    from controller import Supervisor
except ModuleNotFoundError:
    import sys

    sys.exit("This functionality requires webots to be installed")

TIME_STEP = 10

supervisor = Supervisor()  # create Supervisor instance
ego = supervisor.getFromDef('FOLLOWER')
lead = supervisor.getFromDef('LEAD')
# print(dir(ego))
supervisor.simulationResetPhysics()
supervisor.simulationReset()
supervisor.getFromDef("FOLLOWER").restartController()
supervisor.getFromDef("LEAD").restartController()
supervisor.step(TIME_STEP)

i = 0
while supervisor.step(TIME_STEP) != -1:

    # if i % 10 == 1:
    #     print(ego.getPosition())
    i += 1
    # print(i)
    if i > 10000:
        supervisor.simulationReset()
        # supervisor.simulationQuit(0) # or EXIT_FAILURE)
        break

print("[SUPERVISOR] Ended successfully")
