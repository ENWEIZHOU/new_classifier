import numpy as np

if __name__=="__main__":
	a=[[ 0, 1, 2, 3, 4], [ 5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
	print("In the given array[[ 0, 1, 2, 3, 4], [ 5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], the average of the axis 1 is: ")
	print(np.average(a, axis=1))