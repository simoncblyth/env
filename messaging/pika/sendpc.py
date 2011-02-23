"""
For the libary path setup ...

export $(env-runenv-)

"""
import ROOT
ROOT.gSystem.Load("libAbtDataModel")
from aberdeen.DataModel.tests.evs import Evs
evs = Evs()

print evs.ri
e = evs[0]





