#!/usr/bin/env python

#
# An additional algorithm to make histograms.
# The algorithm does the same thing as what the c++ source codes
# dybgaudi/trunk/tutorial/Simulation/SimHistsExample @ release1.0.0-rc01
# do.
#
# usage: nuwa.py -n XX share/simhist.py myhistini
#
#
# ToDo:     1. We don't need all the DybPython.Interactive but part of supplying
#              dictionary for class HepMC::GenEvent
#           2. The bin label associated with detectors has not yet fully completed
#

import GaudiKernel.SystemOfUnits as units
import DybPython.Interactive
from GaudiPython import PyAlgorithm
from GaudiPython import AppMgr
from DybPython.Util import irange
# import these issues to access the geometry info
from GaudiPython.GaudiAlgs import GaudiAlgo
import PyCintex
Gaudi = PyCintex.makeNamespace('Gaudi')

class MyAlg(PyAlgorithm):
    def initialize(self):
        print 'Using PyRoot to make histogram!'
        from ROOT import TCanvas, TFile, TH1F, TH2F, TH1I
        from ROOT import gROOT, gRandom, gSystem, Double
        gROOT.Reset()
        #c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
        self.hfile = TFile( 'pyhsit.root', 'RECREATE', 'Make histograms from TES' )
        self.hpe = TH1F('kineEnergy', 'Particle energy (MeV)', 100,0 ,5 )
        self.ht = TH1F('kineVtxTime','Times of vertices (seconds)',1500,0,15*60)
        self.hpxy = TH2F ('kineVertexXY', 'Primary Vertices, X-Y',100,-3,3,100,-3,3)
        self.hpyz = TH2F ('kineVertexYZ', 'Primary Vertices, Y-Z',100,-3,3,100,-3,3)
        self.hpxz = TH2F ('kineVertexXZ', 'Primary Vertices, Z-X',100,-3,3,100,-3,3)
        self.hkp = TH1F('nKineParts', 'Total number of primaries', 10, 0, 500)
        self.nc = TH1I('nHitCollections','Number of Hit Collections', 15, 0, 15)
        self.hnd = TH2F('nHitByDetector', 'Number of hits in each detector',16, -2,14,100,0,99)
        #self.c1 = c1
        # booking the detector bins. Not a fully complete..............
        dets= {1025:"FarAD1"}

        self.ga = GaudiAlgo("gaalg")

        self.units_energy = units.eV
        self.units_meter = units.meter
        self.units_sec = units.second

        return self.ga.initialize()
#        ga.initialize()
#        return 1

    def execute(self):
        print 'Starting customizing algorithm!'

        print '!!!!!!!!!!!!!!!!units_meter is ', units.meter
        print 'Accessing info from TES.......'
        app = AppMgr()
        esv = app.evtsvc()
        esv.dump()


        ############# Genhist plotting ###########################
        egh = esv['/Event/Gen/GenHeader']
        ee = egh.event()

        epv = ee.signal_process_vertex()
        kpar = epv.particles_out_size()
        self.hkp.Fill(kpar)

        # looping over all particles per event and make histogram
        print 'looping over all particles ...'
        for pax in irange(ee.particles_begin(),ee.particles_end()):
            pme = pax.momentum().e()
            pmeu = pme/self.units_energy # unit:MeV
            self.hpe.Fill(pmeu)

        # accessing the geometry
        det = self.ga.getDet("/dd/Structure/AD/far-oil1")
        print det.name()
        gi = det.geometry()
        # accessing vertex info
        for vtx in irange(ee.vertices_begin(),ee.vertices_end()):
            vtxt = vtx.position().t()
            vtxx, vtxy, vtxz = vtx.position().x(), vtx.position().y(), vtx.position().z()
            vtxts = vtxt/self.units_sec
            gp = Gaudi.XYZPoint(vtxx,vtxy,vtxz)
            lp = gi.toLocal(gp)
            lpx = lp.x()/self.units_meter
            lpy = lp.y()/self.units_meter
            lpz = lp.z()/self.units_meter
            print '!!!!!!!!!!!!!!!!!!!!!',vtxt,vtxts,vtxx,vtxts,lpx

            self.ht.Fill(vtxts)
            self.hpxy.Fill(lpx,lpy)
            self.hpyz.Fill(lpy,lpz)
            self.hpxz.Fill(lpx,lpz)


        ###########  simhist plotting ######################

        sh = esv['/Event/Sim/SimHeader']
        sc = sh.hits().hitCollection()
        scs = sc.size()
        print 'The size of hitCollection is ',sc.size()
        #### Here is /dd/Structure/AD/far-oil1 ##
        import GaudiPython
        det = GaudiPython.ROOT.DayaBay.Detector(0x04,1) # far, AD1
        detsitepd = det.siteDetPackedData()
        print 'detsitepd is ', detsitepd
        # Out[6]: 1025
        #
        self.nc.Fill(scs)
        if scs != 0:
            [i for i in range(detsitepd+1) if sc[i] != None ]
            print 'The index i is ',i
            print 'The size of hitCollection is ',sc.size()
            scv = sc[i]
            scc = scv.collection()
            hitcols = scc.size()
            print '!!!!!!!!!hits no. is ', hitcols
            #### bin labels for a detector. not yet complete for all detector.... ####
            if hitcols == 0:
                print '!!!!!!!!!!!!!!! hitcols is 0!!!!!!!!!!!!!'
                self.hnd.Fill(-1.5,hitcols)
            if detsitepd == 1025:
                self.hnd.Fill(11.5,hitcols)
            else:
                self.hnd.Fill(-0.5,hitcols)
                print '!!!!!!!!!!!hits on other detectors!!!!!!!'
        print '!!!!!!!!!!!!!!!!no hits and scs is ',scs

        return True

    def finalize(self):
#        c1 = self.c1
#        c1.Modified()
#        c1.Update()
        self.hfile.Write()
        del self.hpe
        del self.ht
        del self.hpxy
        del self.hpyz
        del self.hpxz
        del self.hkp
        del self.nc
        del self.hnd
        print 'Histograms Ready!!'
        return True
    def beginRun(self) : return 1
    def endRun(self) : return 1



def run(*args):
    pass

def configure():
#   from GaudiPython import AppMgr
    app = AppMgr()
    print 'Adding customizing algorithm'
    app.addAlgorithm(MyAlg())
    print 'Run!! Go Fight Go!!'
    return
pass

