
from GaudiPython import PyAlgorithm
from GaudiPython import AppMgr
from DybPython.Util import irange

class MyAlg(PyAlgorithm):
	def execute(self):
		print 'Starting customizing algorithm!'

		print 'Using PyRoot to make histogram!'
		from ROOT import TCanvas, TFile, TH1F
		from ROOT import gROOT, gRandom, gSystem, Double
		gROOT.Reset()
		c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
		hfile = TFile( 'thhohsimple.root', 'RECREATE', 'Demo ROOT file with histograms' )
		hpx    = TH1F( 'hpx', 'This is the px distribution', 100, -4, 4 )
		hpxFill = hpx.Fill

		print 'Accessing info from TES.......'
		app = AppMgr()
		esv = app.evtsvc()
		esv.dump()
		egh = esv['/Event/Gen/GenHeader']
		ee = egh.event()
#		print 'looping over all particles ...'
#		count = 0
#		for pax in irange(ee.particles_begin(),ee.particles_end()):
#			print pax
#			print 'counting is ',count
#			count = count + 1
#			hpxFill(pax.momentum().px())

		c1.Modified()
		c1.Update()
		del hpxFill
		hfile.Write()
		print 'Histogram Ready!!'
		return True



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

