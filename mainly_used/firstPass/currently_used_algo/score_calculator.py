import pandas as pd

def score_cal():
	solution = pd.read_table('stage1_solution.csv', sep=',', index_col = 0)
	result = pd.read_table('stage1_submission50_20_0.5.csv', sep=',', index_col = 0)

	solu = solution.cancer.tolist()
	res = result.cancer.tolist()

	total_right = 0
	total_wrong = 0
	right_positive = 0
	right_negetive = 0
	false_positive = 0
	false_negetive = 0

	for ind in range(len(solu)):
		if solu[ind] != res[ind]:
			total_wrong += 1
		else:
			total_right += 1

		if (solu[ind] == 1) and (res[ind] == 0):
			false_negetive += 1
		elif (solu[ind] == 0) and (res[ind] == 1):
			false_positive += 1
		elif (solu[ind] == 1) and (res[ind] == 1):
			right_positive += 1
		elif (solu[ind] == 0) and (res[ind] == 0):
			right_negetive += 1

	print("")
	print("Result: ")
	print("Total number of test dataset is: " + str(len(solu)))
	print("Number of right prediction is: " + str(total_right))
	print("Number of wrong prediction is: " + str(total_wrong))
	print("Number of right positive prediction is: " + str(right_positive))
	print("Number of right negetive prediction is: " + str(right_negetive))
	print("Number of false negetive prediction is: " + str(false_negetive))
	print("Number of false postive prediction is: " + str(false_positive))
	print("Overall accuracy: " + str(total_right / len(solu)))
	print("")


score_cal()
