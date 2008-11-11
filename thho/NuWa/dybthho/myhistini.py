#!/usr/bin/env python

#
# An additional algorithm to make histograms.
# The algorithm does the same thing as what the source codes
# dybgaudi/trunk/tutorial/Simulation/SimHistsExample @ release1.0.0-rc01
# do.
#
# usage: nuwa.py -n XX share/simhist.py myhistini
#
#
# ToDo: 1. The unit seems not correct. Check the C++ codes of tutorial
#       2. We don't need all the DybPython.Interactive but part of supplying
#		   dictionary for class HepMC::GenEvent
#
#

import DybPython.Interactive
from GaudiPython import PyAlgorithm
from GaudiPython import AppMgr
from DybPython.Util import irange

class MyAlg(PyAlgorithm):
	def initialize(self):
		print 'Using PyRoot to make histogram!'
		from ROOT import TCanvas, TFile, TH1F
		from ROOT import gROOT, gRandom, gSystem, Double
		gROOT.Reset()
		c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
		hfile = TFile( 'thhohsimple.root', 'RECREATE', 'Demo ROOT file with histograms' )
		hpx    = TH1F( 'hpx', 'This is the px distribution', 100,0 , 10 )
		self.c1 = c1
		self.hpx = hpx
		self.hfile = hfile
		return True

	def execute(self):
		print 'Starting customizing algorithm!'
		print 'Accessing info from TES.......'
		app = AppMgr()
		esv = app.evtsvc()
		esv.dump()
		egh = esv['/Event/Gen/GenHeader']
		ee = egh.event()
		print 'looping over all particles ...'
		count = 0
		for pax in irange(ee.particles_begin(),ee.particles_end()):
			print pax
			print 'counting is ',count
			pme = pax.momentum().e()
			pmeu = pme * 1000000 # unit:MeV ??
			print 'particle energy is ', pmeu
			count = count + 1
			self.hpx.Fill(pmeu)

         

		return True

	def finalize(self):
		c1 = self.c1
		c1.Modified()
		c1.Update()
		self.hfile.Write()
		del self.hpx
		print 'Histograms Ready!!'
		return True
	def beginRun(self) : return 1
	def endRun(self) : return 1



def run(*args):
	pass

def configure():
#	from GaudiPython import AppMgr
	app = AppMgr()
	print 'Adding customizing algorithm'
	app.addAlgorithm(MyAlg())
	print 'Run!! Go Fight Go!!'
	return
pass

