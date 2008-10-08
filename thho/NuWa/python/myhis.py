from ROOT import TCanvas, TFile, TH1F
from ROOT import gROOT, gRandom, gSystem, Double

gROOT.Reset()

c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )

hfile = TFile( 'thhohsimple.root', 'RECREATE', 'Demo ROOT file with histograms' )

hpx    = TH1F( 'hpx', 'This is the px distribution', 100, -4, 4 )

gRandom.SetSeed()
rannor, rndm = gRandom.Rannor, gRandom.Rndm

hpxFill = hpx.Fill
px, py = Double(), Double()
kkk = 100
for i in xrange( 25000 ):
	rannor( px, py )
	pz = px*px + py*py
	random = rndm(1)
	hpxFill( pz )

	if i and i%kkk == 0:
		if i == kkk:
			hpx.Draw()
			print 'DRAW!!!!!!'

		c1.Modified()
   		c1.Update()

del hpxFill

# Automatically file close
hfile.Write()
