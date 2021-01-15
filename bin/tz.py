#!/usr/bin/env python


offset = 8 

tfmt = "%0.2d:00"
fmt = tfmt + " " + tfmt 

print("%5s %5s" % (".", "+%d" % offset)) 
for i in range(24):
    print(fmt % (i, (i+offset)%24))   

