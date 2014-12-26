#!/usr/bin/env python
import os

# absolute setting is obnoxious, as stomps on override environment vars
# so use setdefault
#
os.environ.setdefault('SQLITE3_DATABASE',"/usr/local/env/nuwa/mocknuwa.db")
import numpy as np
from  _npar import npar as q 

import matplotlib.pyplot as plt

def scatter(sql):
    a = q(sql)
    plt.scatter(a[:,0],a[:,1])
    plt.show()
 
if __name__ == '__main__':
    scatter("select id, tottime from log ;")

