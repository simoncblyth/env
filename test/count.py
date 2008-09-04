import time
import sys


sys.stderr.write("count starting \n")
for i in range(int(sys.argv[1])):
    time.sleep(0.2)


    if i == 5:
        xsg = "... FATAL"
    elif i == 7:
        xsg = "... ERROR"
    else:
        xsg = ""
    msg = "counting %s %s \n" % ( i , xsg ) 

    sys.stdout.write(msg)
    sys.stdout.flush()

sys.stderr.write("count done\n")



