from GaudiPython import PyAlgorithm

class MyAlg(PyAlgorithm):
    def execute(self):
#	print gentools.gun.Momentum
#	import myhis
	from ROOT import TCanvas, TFile, TH1F
	from ROOT import gROOT, gRandom, gSystem, Double
	
	gROOT.Reset()
	
	c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
	
	hfile = TFile( 'thhohsimple.root', 'RECREATE', 'Demo ROOT file with histograms' )
	
	hpx    = TH1F( 'hpx', 'This is the px distribution', 100, 0, 20000 )
	
	
	hpxFill = hpx.Fill
	hpxFill(gentools.gun.Momentum)
	
	c1.Modified()
	c1.Update()
	
	del hpxFill
	
	# Automatically file close
	hfile.Write()
	
	
	print "OOOOOOOOOOOOOOOOOKKKKKKKKKKKKKKKKKKKKKKKKK"
        return True

if __name__=="__main__":
    import gentools
    from GaudiPython import AppMgr
    app = AppMgr()
    app.addAlgorithm(MyAlg())
#    app.run(10) 

