#!/usr/bin/env python

#
# usage: ./myalg.py [config1.py config2.py ...]
#

from GaudiPython import PyAlgorithm
import GaudiKernel.SystemOfUnits as units

class MyAlg(PyAlgorithm):
    def execute(self):
	print 'Starting customizing algorithm!'
        return True

if __name__=="__main__":

        print 'Using the nuwa configuration'
        from DybPython.Control import main
        nuwa = main()

	# Dont use nuwa.run(), using the run configuration to add
	# self defining alforithm
        from GaudiPython import AppMgr
        app = AppMgr()
        print 'Adding customizing algorithm'
        app.addAlgorithm(MyAlg())   
        print 'Run!! Go Fight Go!!'
        app.EvtMax = nuwa.opts.executions
        app.run(app.EvtMax)

	pass #end
