# notifymq-ipython tests/mq_monitor.py
import ROOT
ROOT.gSystem.Load("libnotifymq")
ROOT.gSystem.Load("libAbtDataModel")

from ROOT import gMQ
gMQ.Create(True)      # starts the monitor thread  

# on receiving messages the demo observer methods are invoked...
# note all messages will be observed which may not be desirable, see evmq.py for
# alternative that controls update frequency by only checking on a schedule
