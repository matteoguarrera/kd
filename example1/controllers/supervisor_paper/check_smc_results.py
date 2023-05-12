import numpy as np
import matplotlib.pyplot as plt

monolithic = []
compositional = []
s = np.genfromtxt("halton/smc_csvs/scenario_results.csv", delimiter=",", names=True)
s1 = np.genfromtxt("halton/smc_csvs/subscenario1_results.csv", delimiter=",", names=True)
s2L = np.genfromtxt("halton/smc_csvs/subscenario2L_results.csv", delimiter=",", names=True)
s2S = np.genfromtxt("halton/smc_csvs/subscenario2S_results.csv", delimiter=",", names=True)
s2R = np.genfromtxt("halton/smc_csvs/subscenario2R_results.csv", delimiter=",", names=True)
# monolithic.append(s["sim_steps"].sum())
# compositional.append(s1["sim_steps"].sum() + s2L["sim_steps"].sum() + s2S["sim_steps"].sum() + s2R["sim_steps"].sum())
# print("---- Sample ----")
# print("Monolithic:", s["sim_steps"].sum())
# print("Compositional:", s1["sim_steps"].sum() + s2L["sim_steps"].sum() + s2S["sim_steps"].sum() + s2R["sim_steps"].sum())
# print("-----------------")

# print("Monolithic:", np.mean(monolithic), np.std(monolithic))
# print("Compositional:", np.mean(compositional), np.std(compositional))
print("scenario:", s["sim_steps"].mean(), s["sim_steps"].size)
print("subscenario1:", s1["sim_steps"].mean(), s1["sim_steps"].size)
print("subscenario2L:", s2L["sim_steps"].mean(), s2L["sim_steps"].size)
print("subscenario2S:", s2S["sim_steps"].mean(), s2S["sim_steps"].size)
print("subscenario2R:", s2R["sim_steps"].mean(), s2R["sim_steps"].size)
ss = np.concatenate([s1["sim_steps"], s2L["sim_steps"], s2S["sim_steps"], s2R["sim_steps"]], axis=0)
print("Average:", ss.mean(), ss.size)



# s1 = np.genfromtxt("smc_csvs" + "/subscenario1_post_conditions.csv", delimiter=",", skip_header=True)
# s2L = np.genfromtxt("smc_csvs" + "/subscenario2L_post_conditions.csv", delimiter=",", skip_header=True)
# s2S = np.genfromtxt("smc_csvs" + "/subscenario2S_post_conditions.csv", delimiter=",", skip_header=True)
# s2R = np.genfromtxt("smc_csvs" + "/subscenario2R_post_conditions.csv", delimiter=",", skip_header=True)

# s = np.genfromtxt("smc_csvs" + "/scenario_post_conditions.csv", delimiter=",", skip_header=True)
# # s = s2S
# sems = []
# delta_sems = []
# previous_sem = None
# for i in range(2, np.size(s, axis=0)):
# 	sem = np.std(s[:i], ddof=1, axis=0) / np.sqrt(np.size(s[:i], axis=0))
# 	# print("SEM:", sem)
# 	# print("Rounded:", np.round(sem, 5))
# 	sems.append(sem)
# 	if previous_sem is not None:
# 		delta_sems.append(abs(sem - previous_sem))
# 	# if previous_sem is not None:
# 	# 	print(abs(sem - previous_sem)/previous_sem)
# 	# if previous_sem is not None and np.all(abs(sem - previous_sem)/previous_sem <= 0.01):
# 	# # if previous_sem is not None and np.all(abs(sem - previous_sem) <= 0.001):
# 	# 	print(i, "BREAK")
# 	# 	sp = np.genfromtxt("smc_csvs_Feb_1" + "/scenario_results.csv", delimiter=",", names=True)
# 	# 	# sp = np.genfromtxt("smc_csvs" + "/subscenario1_results.csv", delimiter=",", names=True)
# 	# 	print(sp[:i]["sim_steps"].sum())
# 	# 	break
# 	previous_sem = sem
# sems = np.array(sems)
# # sems = sems / np.max(sems, axis=0)
# plt.plot(delta_sems[300:])
# plt.show()
