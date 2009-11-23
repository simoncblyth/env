#   notifymq-ipython tests/mq_sendobject.py 
import ROOT
ROOT.gSystem.Load("lib/libnotifymq.so")
ROOT.MQ.Create()

ROOT.gMQ.SendString( "hello" )

from aberdeen.DataModel.tests.evs import Evs
evs = Evs()
ROOT.gMQ.SendObject( evs[100] )


