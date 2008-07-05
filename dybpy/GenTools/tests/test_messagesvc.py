

import ROOT
ROOT.gSystem.Load("libMathCore")  
import GaudiPython as gp 
import PyCintex as pc
## 4 lines needed to come at the beginning for streams for work ???


g = gp.AppMgr()

"""

  this has some degree of working but crashes...

ims = g.service("MessageSvc",interface="IMessageSvc")
import sys
path = "%s.log" % sys.modules[__name__].__file__
#path = "/dev/stdout"
log = gp.gbl.ofstream(path)
ims.setDefaultStream(log)

"""



"""
   follow example from 
      gaudi/GaudiPython/tests/test_basics.py
      
    gaining python control of the message stream
"""


def echo(s):print "echo..%s" % s
buf = gp.CallbackStreamBuf(echo)
ost = gp.gbl.ostream(buf)
msv = g.service("MessageSvc", "IMessageSvc")
ori = msv.defaultStream()

msv.setDefaultStream(ost)
msv.reportMessage('TEST',7,'This is a test message')
g.initialize()
msv.setDefaultStream(ori)




"""
In [14]: g.initialize()
echo..HistogramPersis...   INFO  'CnvServices':[ 'HbookHistSvc' , 'RootHistSvc' ]

echo..HistogramPersis...WARNING Histograms saving not required.

echo..ApplicationMgr       INFO Application Manager Initialized successfully

Out[14]: SUCCESS


In [15]: o = g.initialize()
echo..ApplicationMgr       INFO Already Initialized!

In [16]: o
Out[16]: SUCCESS



"""





