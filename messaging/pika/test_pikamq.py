#!/usr/bin/env python

from pikamq import PikaMQ

if __name__ == '__main__':
    pmq = PikaMQ()
    pmq()

    #import ROOT
    #ROOT.gSystem.Load("libAbtDataModel")
    #from aberdeen.DataModel.tests.evs import Evs

    ## sending pickled ROOT objects, working OK 
    #evs = Evs()
    #p.put( pickle.dumps(evs.ri) )
    #p.put( pickle.dumps(evs[0]) )
    #p.put( pickle.dumps(evs[1]) )



