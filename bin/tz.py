#!/usr/bin/env python
"""
                                          SFO     LHR     PEK
                                        16:00   00:00   08:00
                                        17:00   01:00   09:00
                                        18:00   02:00   10:00
                                        19:00   03:00   11:00
                                        20:00   04:00   12:00
                                        21:00   05:00   13:00
                                        22:00   06:00   14:00
                                        23:00   07:00   15:00
                                        00:00   08:00   16:00
                                        01:00   09:00   17:00
                                        02:00   10:00   18:00
                                        03:00   11:00   19:00
                                        04:00   12:00   20:00
                                        05:00   13:00   21:00
                                        06:00   14:00   22:00
                                        07:00   15:00   23:00
                                        08:00   16:00   00:00
                                        09:00   17:00   01:00
                                        10:00   18:00   02:00
                                        11:00   19:00   03:00
                                        12:00   20:00   04:00
                                        13:00   21:00   05:00
                                        14:00   22:00   06:00
                                        15:00   23:00   07:00
                                          SFO     LHR     PEK

"""
import numpy as np

labels = ["SFO","LHR","PEK"]
offsets = [-8, 0, 8]
assert len(labels) == len(offsets)

h24 = np.arange(0, 24, dtype=np.int32 ) 
hrs = np.zeros( [len(offsets),24], dtype=np.int32 ) 

for i in range(len(offsets)):
    hrs[i] = (h24 + offsets[i])%24
pass

tfmt1, lfmt1, div = "%0.2d:00", "%5s", "   " 
tfmt = div.join([tfmt1 for _ in range(len(hrs))]) 
lfmt = div.join([lfmt1 for _ in range(len(hrs))]) 

pfx = " " * 40 
print(pfx + lfmt % tuple(labels))
print("\n".join([ pfx + tfmt % tuple(map(int,_)) for _ in hrs.T] ))
print(pfx + lfmt % tuple(labels))

