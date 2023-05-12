from verifai.samplers.scenic_sampler import ScenicSampler
from verifai.scenic_server import ScenicServer
from dotmap import DotMap
from verifai.falsifier import mtl_falsifier, generic_falsifier
from verifai.features.features import *
from verifai.monitor import specification_monitor, mtl_specification
from time import sleep
import pickle
import random
import math
import sys
import os

PORT = 8888
MAXREQS = 5
BUFSIZE = 4096

MODE = sys.argv[1]
assert MODE in ["falsify", "smc"]
print("Mode:", MODE)

METHOD = sys.argv[2]
assert METHOD in ["compositional", "monolithic"]
print("Method:", METHOD)

BATCH_SIZE = None
try:
    BATCH_SIZE = int(sys.argv[3])
    assert BATCH_SIZE > 0
    print("Batch Size:", BATCH_SIZE)
except:
    BATCH_SIZE = None
    print("Running until convergence...")

with open("mode.txt", "w+") as f:
    f.write(MODE)

def get_falsifier(scenario, specification):

    path = os.path.join(os.path.dirname(__file__), scenario)
    sampler = ScenicSampler.fromScenario(path)

    falsifier_params = DotMap(
        n_iters=BATCH_SIZE,
        max_time=None,
        save_error_table=True,
        save_safe_table=True
    )

    server_options = DotMap(
        port=PORT,
        bufsize=BUFSIZE,
        maxreqs=MAXREQS
    )

    falsifier = mtl_falsifier(
        sampler=sampler,
        sampler_type="halton",
        specification=specification,
        falsifier_params=falsifier_params,
        server_options=server_options
    )

    return falsifier

# Assumption: A subscenario can only be in one subscenario group
subscenario_groups = None
if METHOD == "compositional":
    subscenario_groups = [["subscenario1"], ["subscenario2L", "subscenario2S", "subscenario2R"]]
else:
    subscenario_groups = [["scenario"]]

n_unique_subscenarios = len([subscenario for subscenario_group in subscenario_groups for subscenario in subscenario_group])

for subscenario_group in subscenario_groups:
    for subscenario in subscenario_group:
        with open(MODE + "_csvs/" + subscenario + "_post_conditions.csv", "+w") as f:
            f.write("ego_x,ego_y,ego_heading,lead_x,lead_y,lead_heading\n")
        with open(MODE + "_csvs/" + subscenario + "_results.csv", "+w") as f:
            f.write("sample_ind,rho,sim_steps\n")
        with open(MODE + "_csvs/" + subscenario + "_samples.csv", "+w") as f:
            f.write("sample_ind,sample\n")
# sys.exit()

specification = ["G(distance)"]

if MODE == "falsify":
    if BATCH_SIZE is None:
        for subscenario_group in subscenario_groups:
            subscenario_group_shuffled = random.sample(subscenario_group, len(subscenario_group))
            for subscenario in subscenario_group_shuffled:
                falsifier = get_falsifier(subscenario + ".scenic", specification)
                try:
                    print("Falsifier is running for " + subscenario + "...")
                    falsifier.run_falsifier()
                except EOFError:
                    print("Falsification ended successfully.")
                    print("Sleeping for 60 seconds...")
                    sleep(60)
                    sys.exit()
                except ValueError:
                    print("Scenario/Sub-scenario converged.")
                print("Done.")
                print("Sleeping for 60 seconds...")
                sleep(60)
    else:
        i = 0
        converged_subscenarios = []
        while True:
            i += 1
            print("Batched Iteration", i)
            for subscenario_group in subscenario_groups:
                subscenario_group_shuffled = random.sample(subscenario_group, len(subscenario_group))
                for subscenario in subscenario_group_shuffled:
                    if subscenario not in converged_subscenarios:
                        falsifier = get_falsifier(subscenario + ".scenic", specification)
                        try:
                            print("Falsifier is running for " + subscenario + "...")
                            falsifier.run_falsifier()
                        except EOFError:
                            print("Falsification ended successfully.")
                            print("Sleeping for 60 seconds...")
                            sleep(60)
                            sys.exit()
                        except ValueError:
                            print("Scenario/Sub-scenario converged.")
                            converged_subscenarios.append(subscenario)
                        print("Done.")
                        print("Sleeping for 60 seconds...")
                        sleep(60)
            if len(converged_subscenarios) == n_unique_subscenarios:
                break
    print("All scenarios/sub-scenarios converged.")
    print("The system could not be falsified.")
else:
    for subscenario_group in subscenario_groups:
        subscenario_group_shuffled = random.sample(subscenario_group, len(subscenario_group))
        for subscenario in subscenario_group_shuffled:
            falsifier = get_falsifier(subscenario + ".scenic", specification)
            try:
                print("Falsifier is running for " + subscenario + "...")
                falsifier.run_falsifier()
            except ValueError as e:
                print("Scenario/Sub-scenario converged.")
            print("Done.")
            print("Sleeping for 60 seconds...")
            sleep(60)
