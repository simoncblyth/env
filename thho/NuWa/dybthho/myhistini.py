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
#          dictionary for class HepMC::GenEvent
#
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
        from ROOT import TCanvas, TFile, TH1F, TH2F
        from ROOT import gROOT, gRandom, gSystem, Double
        gROOT.Reset()
#        c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
        hfile = TFile( 'thhohsimple.root', 'RECREATE', 'Make histograms from TES' )
        m_hpe    = TH1F( 'hpe', 'Particle energy (MeV)', 100,0 ,5 )
        m_hpxy = TH2F ('hpxy', 'Primary Vertices, X-Y',100,-3,3,100,-3,3)
        m_hpyz = TH2F ('hpyz', 'Primary Vertices, Y-Z',100,-3,3,100,-3,3)
        m_hpxz = TH2F ('hpxz', 'Primary Vertices, Z-X',100,-3,3,100,-3,3)
#        self.c1 = c1
        self.hfile = hfile
        self.hpe = m_hpe
        self.hpxy = m_hpxy
        self.hpyz = m_hpyz
        self.hpxz = m_hpxz

        ga = GaudiAlgo("MyAlg")
        self.ga = ga

        return ga.initialize()

    def execute(self):
        print 'Starting customizing algorithm!'
        units_energy = units.eV
        units_meter = units.meter
        print '!!!!!!!!!!!!!!!!units_meter is ', units.meter
        print 'Accessing info from TES.......'
        app = AppMgr()
        esv = app.evtsvc()
        esv.dump()
        egh = esv['/Event/Gen/GenHeader']
        ee = egh.event()

        # looping over all particles per event and make histogram
        print 'looping over all particles ...'
        for pax in irange(ee.particles_begin(),ee.particles_end()):
            pme = pax.momentum().e()
            pmeu = pme/units_energy # unit:MeV
            self.hpe.Fill(pmeu)

        # accessing the geometry
        det = self.ga.getDet("/dd/Structure/AD/far-oil1")
        print det.name()
        gi = det.geometry()
        # accessing vertex info
        for vtx in irange(ee.vertices_begin(),ee.vertices_end()):
            vtxx = vtx.position().px()
            vtxy = vtx.position().py()
            vtxz = vtx.position().pz()
#            print vtxx
            print '!!!!!!!!!!!!!!!!!!!!!!!vtxx is ', vtxx
            print '!!!!!!!!!!!!!!!!!!!!!!!vtxy is ', vtxy
            print '!!!!!!!!!!!!!!!!!!!!!!!vtxz is ', vtxz
            vtxy = vtx.position().py()
            vtxz = vtx.position().pz()
            vtxxm = vtxx/units_meter
            vtxym = vtxy/units_meter
            vtxzm = vtxz/units_meter
            gp = Gaudi.XYZPoint(vtxx,vtxy,vtxz)
            lp = gi.toLocal(gp)
            print '!!!!!!!!!!!!!!!!!!!!!!!lpx is ', lp.x()
            print '!!!!!!!!!!!!!!!!!!!!!!!lpy is ', lp.y()
            print '!!!!!!!!!!!!!!!!!!!!!!!lpz is ', lp.z()

#            self.hpxy.Fill(vtxxm,vtxym)
#            self.hpyz.Fill(vtxym,vtxzm)
#            self.hpxz.Fill(vtxx,vtxz)



        return True

    def finalize(self):
#        c1 = self.c1
#        c1.Modified()
#        c1.Update()
        self.hfile.Write()
        del self.hpe
        del self.hpxy
        del self.hpyz
        del self.hpxz
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

