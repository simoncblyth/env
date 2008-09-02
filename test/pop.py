
import os
import string

def cmd( cmdline ):
    for line in os.popen(cmdline).readlines():    
        pos = string.find(line, "FATAL" )
        print "[%-3s] %s " % (pos , line ),            

if __name__=='__main__':
    cmd("python share/geniotest.py")
    cmd("python share/geniotest.py input")



