import traceback 
import sys

# http://www.lightbird.net/py-by-example/traceback-module.html


def x1():
	try: 1/0
	except:
		t,v,tb = sys.exc_info()
		traceback.print_exception(t,v,tb)

def x2():

	try: 1/0
    	except:
        	tb = sys.exc_info()[2]
        	lst = traceback.format_list(traceback.extract_stack())
        	for l in lst: print l,





#x1()
x2()


