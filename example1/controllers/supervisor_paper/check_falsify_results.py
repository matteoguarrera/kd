import numpy as np

monolithic = []
compositional = []
for i in range(1, 11):
	s = np.genfromtxt("halton/falsify_csvs_" + str(i) + "/scenario_results.csv", delimiter=",", names=True)
	s1 = np.genfromtxt("halton/falsify_csvs_" + str(i) + "/subscenario1_results.csv", delimiter=",", names=True)
	s2L = np.genfromtxt("halton/falsify_csvs_" + str(i) + "/subscenario2L_results.csv", delimiter=",", names=True)
	s2S = np.genfromtxt("halton/falsify_csvs_" + str(i) + "/subscenario2S_results.csv", delimiter=",", names=True)
	s2R = np.genfromtxt("halton/falsify_csvs_" + str(i) + "/subscenario2R_results.csv", delimiter=",", names=True)
	monolithic.append(s["sim_steps"].sum())
	compositional.append(s1["sim_steps"].sum() + s2L["sim_steps"].sum() + s2S["sim_steps"].sum() + s2R["sim_steps"].sum())
	print("---- Sample", i, "----")
	print("Monolithic:", s["sim_steps"].sum())
	print("Compositional:", s1["sim_steps"].sum() + s2L["sim_steps"].sum() + s2S["sim_steps"].sum() + s2R["sim_steps"].sum())
	print("-----------------")

print("Monolithic:", np.mean(monolithic), np.std(monolithic))
print("Compositional:", np.mean(compositional), np.std(compositional))