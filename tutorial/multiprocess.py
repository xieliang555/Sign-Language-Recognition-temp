from multiprocessing import Process
import os


def task():
	print("task")
	print("当前线程： {}, 父线程： {}\n".format(os.getpid(), os.getppid()))


if __name__ =="__main__":

	t = Process(target=task)
	t.start()
	# 一个线程中调用另一个线程的join方法，调用者被阻塞，直到被调用线程终止
	# 一个线程可以被终止多次
	t.join()

	print("main")
	# 当前父进程是sublime进程
	print("当前线程： {}, 父线程： {}\n".format(os.getpid(), os.getppid()))