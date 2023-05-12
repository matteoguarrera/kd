import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 20})

MODE = sys.argv[1]
assert MODE in ["falsify", "smc"]
print("Mode:", MODE)

if MODE == "smc":
	SMC_GRAPH = sys.argv[2]
	assert SMC_GRAPH in ["bar", "line"]
	if SMC_GRAPH == "line":
		SAMPLER_TYPE = sys.argv[3]
		assert SAMPLER_TYPE in ["random", "halton"]


def get_results(sampler_type):
	assert sampler_type in ["random", "halton"]
	if MODE == "falsify":
		monolithic = []
		compositional = []
		for i in range(1, 11):
			s = np.genfromtxt(sampler_type + "/falsify_csvs_" + str(i) + "/scenario_results.csv", delimiter=",", names=True)
			s1 = np.genfromtxt(sampler_type + "/falsify_csvs_" + str(i) + "/subscenario1_results.csv", delimiter=",", names=True)
			s2L = np.genfromtxt(sampler_type + "/falsify_csvs_" + str(i) + "/subscenario2L_results.csv", delimiter=",", names=True)
			s2S = np.genfromtxt(sampler_type + "/falsify_csvs_" + str(i) + "/subscenario2S_results.csv", delimiter=",", names=True)
			s2R = np.genfromtxt(sampler_type + "/falsify_csvs_" + str(i) + "/subscenario2R_results.csv", delimiter=",", names=True)
			monolithic.append(s["sim_steps"].sum())
			compositional.append(s1["sim_steps"].sum() + s2L["sim_steps"].sum() + s2S["sim_steps"].sum() + s2R["sim_steps"].sum())
		return np.mean(monolithic), np.std(monolithic), np.mean(compositional), np.std(compositional)
	else:
		s = np.genfromtxt(sampler_type + "/smc_csvs/scenario_results.csv", delimiter=",", names=True)
		s1 = np.genfromtxt(sampler_type + "/smc_csvs/subscenario1_results.csv", delimiter=",", names=True)
		s2L = np.genfromtxt(sampler_type + "/smc_csvs/subscenario2L_results.csv", delimiter=",", names=True)
		s2S = np.genfromtxt(sampler_type + "/smc_csvs/subscenario2S_results.csv", delimiter=",", names=True)
		s2R = np.genfromtxt(sampler_type + "/smc_csvs/subscenario2R_results.csv", delimiter=",", names=True)
		return s["sim_steps"].sum(), s1["sim_steps"].sum() + s2L["sim_steps"].sum() + s2S["sim_steps"].sum() + s2R["sim_steps"].sum()

def get_post_conditions(sampler_type):
	s = np.genfromtxt(sampler_type + "/smc_csvs/scenario_post_conditions.csv", delimiter=",", skip_header=True)
	s1 = np.genfromtxt(sampler_type + "/smc_csvs/subscenario1_post_conditions.csv", delimiter=",", skip_header=True)
	s2L = np.genfromtxt(sampler_type + "/smc_csvs/subscenario2L_post_conditions.csv", delimiter=",", skip_header=True)
	s2S = np.genfromtxt(sampler_type + "/smc_csvs/subscenario2S_post_conditions.csv", delimiter=",", skip_header=True)
	s2R = np.genfromtxt(sampler_type + "/smc_csvs/subscenario2R_post_conditions.csv", delimiter=",", skip_header=True)
	return s, s1, s2L, s2S, s2R

def get_average_sems(s):
	sems = []
	# delta_sems = []
	# previous_sem = None
	for i in range(2, np.size(s, axis=0)):
		sem = np.std(s[:i], ddof=1, axis=0) / np.sqrt(np.size(s[:i], axis=0))
		sems.append(sem)
		# if previous_sem is not None:
		# 	delta_sems.append(abs(sem - previous_sem))
		# previous_sem = sem
	return np.array(sems).mean(axis=1)

if MODE == "falsify":
	random_monolithic_mean, random_monolithic_std, random_compositional_mean, random_compositional_std = get_results("random")
	halton_monolithic_mean, halton_monolithic_std, halton_compositional_mean, halton_compositional_std = get_results("halton")
	
	monolithic_means = [random_monolithic_mean, halton_monolithic_mean]
	monolithic_stds = [random_monolithic_std, halton_monolithic_std]
	
	compositional_means = [random_compositional_mean, halton_compositional_mean]
	compositional_stds = [random_compositional_std, halton_compositional_std]

	ind = np.arange(len(monolithic_means))  # the x locations for the groups
	width = 0.3  # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind - width/2, monolithic_means, width, yerr=monolithic_stds, label='Monolithic')
	rects2 = ax.bar(ind + width/2, compositional_means, width, yerr=compositional_stds, label='Compositional')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylim([0, 100000])
	ax.ticklabel_format(style='sci', axis='y', scilimits=(6,6))
	ax.set_ylabel('Number of Simulator Steps')
	# ax.set_title('Compositional Falsification -- Case Study 1')
	ax.set_xticks(ind)
	ax.set_xticklabels(("Uniform", "Halton"))
	ax.set_xlabel('Sampling Strategies')
	ax.legend()


	def autolabel(rects, xpos='center'):
	    """
	    Attach a text label above each bar in *rects*, displaying its height.

	    *xpos* indicates which side to place the text w.r.t. the center of
	    the bar. It can be one of the following {'center', 'right', 'left'}.
	    """

	    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
	    offset = {'center': 0, 'right': 1, 'left': -1}

	    for rect in rects:
	        height = rect.get_height()
	        ax.annotate('{}'.format(height),
	                    xy=(rect.get_x() + rect.get_width() / 2, height),
	                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
	                    textcoords="offset points",  # in both directions
	                    ha=ha[xpos], va='bottom')


	autolabel(rects1, "right")
	autolabel(rects2, "right")

	# fig.tight_layout()

	fig.set_figheight(6)
	fig.set_figwidth(10)

	# plt.show()

	plt.savefig("plots/example1_falsify.pdf")
else:
	if SMC_GRAPH == "bar":
		random_monolithic, random_compositional = get_results("random")
		halton_monolithic, halton_compositional = get_results("halton")
		
		monolithic = [random_monolithic, halton_monolithic]
		
		compositional = [random_compositional, halton_compositional]

		ind = np.arange(len(monolithic))  # the x locations for the groups
		width = 0.3  # the width of the bars

		fig, ax = plt.subplots()
		rects1 = ax.bar(ind - width/2, monolithic, width, label='Monolithic')
		rects2 = ax.bar(ind + width/2, compositional, width, label='Compositional')

		# Add some text for labels, title and custom x-axis tick labels, etc.
		ax.ticklabel_format(style='sci', axis='y', scilimits=(6,6))
		ax.set_ylim([0, 2000000])
		ax.set_ylabel('Number of Simulator Steps')
		# ax.set_title('Compositional Statistical Verification -- Case Study 1')
		ax.set_xticks(ind)
		ax.set_xticklabels(("Uniform", "Halton"))
		ax.set_xlabel('Sampling Strategies')
		ax.legend()


		def autolabel(rects, xpos='center'):
		    """
		    Attach a text label above each bar in *rects*, displaying its height.

		    *xpos* indicates which side to place the text w.r.t. the center of
		    the bar. It can be one of the following {'center', 'right', 'left'}.
		    """

		    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
		    offset = {'center': 0, 'right': 1, 'left': -1}

		    for rect in rects:
		        height = rect.get_height()
		        ax.annotate('{}'.format(height),
		                    xy=(rect.get_x() + rect.get_width() / 2, height),
		                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
		                    textcoords="offset points",  # in both directions
		                    ha=ha[xpos], va='bottom')


		autolabel(rects1, "center")
		autolabel(rects2, "center")

		# fig.tight_layout()

		fig.set_figheight(6)
		fig.set_figwidth(10)

		# plt.show()
		plt.savefig("plots/example1_smc_bar.pdf")
	else:
		if SAMPLER_TYPE  == "random":

			random_s, random_s1, random_s2L, random_s2S, random_s2R = get_post_conditions("random")

			random_s_sems = get_average_sems(random_s)
			random_s1_sems = get_average_sems(random_s1)
			random_s2L_sems = get_average_sems(random_s2L)
			random_s2S_sems = get_average_sems(random_s2S)
			random_s2R_sems = get_average_sems(random_s2R)

			plt.figure(figsize=(10,6))

			plt.plot(np.arange(len(random_s_sems)), random_s_sems, label ='Monolithic scenario')
			plt.plot(np.arange(len(random_s1_sems)), random_s1_sems, label ='Compositional subScenario1')
			plt.plot(np.arange(len(random_s2L_sems)), random_s2L_sems, label ='Compositional subScenario2L')
			plt.plot(np.arange(len(random_s2S_sems)), random_s2S_sems, label ='Compositional subScenario2S')
			plt.plot(np.arange(len(random_s2R_sems)), random_s2R_sems, label ='Compositional subScenario2R')

			plt.xlabel("Number of Simulations")
			plt.ylabel("Standard Error of the Mean")
			plt.legend()
			# plt.title('Convergence of SEM with Random Sampling -- Case Study 1')
			plt.savefig("plots/example1_smc_line_random.pdf")
		else:

			halton_s, halton_s1, halton_s2L, halton_s2S, halton_s2R = get_post_conditions("halton")
			halton_s_sems = get_average_sems(halton_s)
			halton_s1_sems = get_average_sems(halton_s1)
			halton_s2L_sems = get_average_sems(halton_s2L)
			halton_s2S_sems = get_average_sems(halton_s2S)
			halton_s2R_sems = get_average_sems(halton_s2R)

			plt.figure(figsize=(10,6))

			plt.plot(np.arange(len(halton_s_sems)), halton_s_sems, label ='Monolithic scenario')
			plt.plot(np.arange(len(halton_s1_sems)), halton_s1_sems, label ='Compositional subScenario1')
			plt.plot(np.arange(len(halton_s2L_sems)), halton_s2L_sems, label ='Compositional subScenario2L')
			plt.plot(np.arange(len(halton_s2S_sems)), halton_s2S_sems, label ='Compositional subScenario2S')
			plt.plot(np.arange(len(halton_s2R_sems)), halton_s2R_sems, label ='Compositional subScenario2R')

			plt.xlabel("Number of Simulations")
			plt.ylabel("Standard Error of the Mean")
			plt.legend()
			# plt.title('Convergence of SEM with Halton Sampling -- Case Study 1')
			plt.savefig("plots/example1_smc_line_halton.pdf")

