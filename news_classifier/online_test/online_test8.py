import pandas as pd

def method1():
	#use loop for count
	data=pd.read_table("test_truth.txt", sep='\t', header=None,names=["test","truth"])
	data["check"]=data.test+data.truth
	count=0
	for i in range(0,1000):
		if data["check"][i]==0 or data["check"][i]==2:
			count+=1
	print(count)
def method2():
	#use list map for count
	data=pd.read_table("test_truth.txt", sep='\t', header=None,names=["test","truth"])
	data["check"]=data.test+data.truth
	check=list(map(int,data["check"]))
	print(check.count(0)+check.count(2))

if __name__=="__main__":
	method1()

