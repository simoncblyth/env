# notifymq-ipython tests/mq_monitor.py
import ROOT
ROOT.gSystem.Load("lib/libnotifymq.so")
ROOT.gSystem.Load("$ABERDEEN_HOME/DataModel/lib/libAbtDataModel.so")

from ROOT import gMQ
gMQ.Create(True)
while not(gMQ.IsMonitorFinished()):
    if gMQ.IsBytesUpdated():
        obj = gMQ.ConstructObject()
        if obj:obj.Print()
    ROOT.gSystem.Sleep(100)
    ROOT.gSystem.ProcessEvents() 

