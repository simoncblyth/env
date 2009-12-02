# notifymq-ipython tests/mq_monitor.py
import ROOT
ROOT.gSystem.Load("lib/libnotifymq.so")
ROOT.gSystem.Load("$ABERDEEN_HOME/DataModel/lib/libAbtDataModel.so")

from ROOT import gMQ
gMQ.Create(True)

