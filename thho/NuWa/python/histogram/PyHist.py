#!/usr/bin/env python

'''
An additional algorithm to make histograms.
This python script will make histograms as what the c++ of package tutorial/SimHistsExample does.

usage: nuwa.py -n XX share/simhist.py SimHistsExample.PyHist

'''

from GaudiPython import PyAlgorithm
from GaudiPython import AppMgr
from DybPython.Util import irange
import GaudiPython as gp
import PyCintex
import GaudiKernel.SystemOfUnits as units
import DybPython.Interactive
Gaudi = PyCintex.makeNamespace('Gaudi')

class detSiteDidMap:
    '''
    Establishing a "map" of Site and DetectorId with dict
    '''
    def __init__(self):

        deti = gp.ROOT.DayaBay.Detector
        sis = gp.gbl.Site.FromString
        di = gp.gbl.DetectorId

        detsdm = [deti(sis("DayaBay"),di.kAD1),
                  deti(sis("DayaBay"),di.kAD2),
                  deti(sis("DayaBay"),di.kOWS),
                  deti(sis("DayaBay"),di.kIWS),
                  deti(sis("DayaBay"),di.kRPC),
                  deti(sis("LingAo"),di.kAD1),
                  deti(sis("LingAo"),di.kAD2),
                  deti(sis("LingAo"),di.kOWS),
                  deti(sis("LingAo"),di.kIWS),
                  deti(sis("LingAo"),di.kRPC),
                  deti(sis("Far"),di.kAD1),
                  deti(sis("Far"),di.kAD2),
                  deti(sis("Far"),di.kAD3),
                  deti(sis("Far"),di.kAD4),
                  deti(sis("Far"),di.kOWS),
                  deti(sis("Far"),di.kIWS),
                  deti(sis("Far"),di.kRPC),
                 ]

        detbins = {}
        detsitelist = []
        for ind in range(17):
            detbins[detsdm[ind].siteDetPackedData()] = ind+1
            detsitelist.append(detsdm[ind].siteDetPackedData())

        self.deti = deti
        self.sis = sis
        self.di = di
        self.detsdm = detsdm
        self.detbins = detbins
        self.detsitelist = detsitelist

        pass


class MyAlg(PyAlgorithm):

    '''
    Translation of src/GenHists.cc and src/SimHists.cc histograms making by using PyAlgorithm

    '''
    def initialize(self):
        print 'Using PyRoot to make histogram!'
        from ROOT import TCanvas, TFile, TH1F, TH2F, TH1I
        from ROOT import gROOT, gRandom, gSystem, Double
        #gROOT.Reset()
        #c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
        self.hfile = TFile( 'pyhists.root', 'RECREATE', 'Make histograms from TES' )
        self.hpe = TH1F('kineEnergy', 'Particle energy (MeV)', 100,0 ,5 )
        self.ht = TH1F('kineVtxTime','Times of vertices (seconds)',1500,0,15*60)
        self.hpxy = TH2F ('kineVertexXY', 'Primary Vertices, X-Y',100,-3,3,100,-3,3)
        self.hpyz = TH2F ('kineVertexYZ', 'Primary Vertices, Y-Z',100,-3,3,100,-3,3)
        self.hpxz = TH2F ('kineVertexXZ', 'Primary Vertices, Z-X',100,-3,3,100,-3,3)
        self.hkp = TH1F('nKineParts', 'Total number of primaries', 10, 0, 500)
        self.nc = TH1I('nHitCollections','Number of Hit Collections', 15, 0, 15)
        self.hnd = TH2F('nHitByDetector', 'Number of hits in each detector',16, -2,14,100,0,99)
        #self.c1 = c1
        # booking the detector bins. Not complete
        #dets= {1025:"FarAD1"}

        self.units_energy = units.eV
        self.units_meter = units.meter
        self.units_sec = units.second

        self.dsdm = detSiteDidMap()

        return 1

    def execute(self):
        print 'Executing customizing algorithm......'
        app = AppMgr()
        esv = app.evtsvc()
        #esv.dump()


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
        dsv = app.detSvc()
        det = dsv['/dd/Structure/AD/far-oil1']
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

            self.ht.Fill(vtxts)
            self.hpxy.Fill(lpx,lpy)
            self.hpyz.Fill(lpy,lpz)
            self.hpxz.Fill(lpx,lpz)


        ###########  simhist plotting ######################

        sh = esv['/Event/Sim/SimHeader']
        sc = sh.hits().hitCollection()
        scs = sc.size()
        self.nc.Fill(scs)
        if scs != 0:
            for detsitepd in self.dsdm.detsitelist:
                try:
                    kdet = [i for i in range(detsitepd+1) if sc[i] != None ]
                    for kd in kdet:
                        scv = sc[kd]
                        scc = scv.collection()
                        hitcols = scc.size()
                        bin = self.dsdm.detbins[kd]
                        self.hnd.Fill(bin+0.5, hitcols)
                        assert len(kdet) == scs
                except IndexError:
                    pass

        return True

    def finalize(self):
        #c1 = self.c1
        #c1.Modified()
        #c1.Update()
        self.hfile.Write()
        del self.hpe
        del self.ht
        del self.hpxy
        del self.hpyz
        del self.hpxz
        del self.hkp
        del self.nc
        del self.hnd
        return True
    def beginRun(self) : return 1
    def endRun(self) : return 1



def run(*args):
    pass

def configure():
    #from GaudiPython import AppMgr
    app = AppMgr()
    print 'Adding customizing histogram algorithm'
    app.addAlgorithm(MyAlg())
    print 'Customizing histogram algorithm gets ready to use'
    return
pass

def checkSitepd():
    #name = self.deti(1<<k,j).detName()
    #return self.deti.siteDetPackedFromString(name)
    pass

if '__main__' == __name__:
    print __doc__
