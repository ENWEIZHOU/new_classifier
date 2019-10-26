import sys
def fib(n):
	fib=[0,1]
	if n>2:
		while (len(fib)<n+1):
			fib.append(fib[-1]+fib[-2])
		print("The", n, "-th entry of the Fibonacci series is",fib[-1])
	else:
		if n<1:
			print("Invalid input of the n-th entry.")
		else:
			print("The", n ,"-th entry of the Fibonacci series is", fib[n-1])

if __name__=="__main__":
	print("Please enter the n-th(n>0) entry of the Fibonacci series:")
	n_th=sys.stdin.readline().strip()
	n_th=list(map(int,n_th.split()))
	fib(n_th[0])