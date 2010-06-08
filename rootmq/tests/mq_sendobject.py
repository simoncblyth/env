#   rootmq-ipython tests/mq_sendobject.py 
import ROOT
ROOT.gSystem.Load("librootmq")
ROOT.MQ.Create()
from ROOT import gMQ
gMQ.SendString( "hello from mq_sendobject.py" )

from aberdeen.DataModel.tests.evs import Evs
evs = Evs()
gMQ.SendObject( evs[100] , "abt.test.event" )


