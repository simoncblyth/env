
import traceback
import sys

def tb_filename(tb):
    return tb.tb_frame.f_code.co_filename

def tb_iter(tb):
    while tb is not None:
        yield tb
        tb = tb.tb_next

def alpha():
    try:
        beta()
    except Exception, e:
        etype, value, tb = sys.exc_info()
        filename = tb_filename(tb)
        for tb in tb_iter(tb):
			tbf = tb_filename(tb)
			print tbf
			if tbf != filename:
				break
	traceback.print_exception(etype, value, tb)

def beta():
    gamma()

def gamma():
    exec s in {}

s = """
def delta():
    epsilon()

def epsilon():
    print hi
delta()
"""

if __name__ == "__main__":
    alpha()