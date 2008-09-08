import time
import sys


def msg(i):
    if i == 3: 
        xsg = "#1 "	
    elif i == 5:
        xsg = "... FATAL"
    elif i == 7:
        xsg = "... ERROR"
    elif i == 8:
        xsg = " *** Break *** segmentation violation"
    else:
        xsg = ""
    return "%s %s \n" % ( xsg , i  ) 
   


sys.stderr.write("count starting \n")
for i in range(int(sys.argv[1])):
    time.sleep(0.2)
    sys.stdout.write(msg(i))
    sys.stdout.flush()

sys.stderr.write("count done\n")



