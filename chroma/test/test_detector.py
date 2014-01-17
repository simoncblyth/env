#!/usr/bin/env python
"""
Investigate failing test, from /usr/local/env/chroma_env/src/chroma/test/test_detector.py

Finding that sometimes succeeds, probably just 1000 trials is not enough stats
to be sure of the outcome of the propagation::

    INFO:__main__:testCharge
    INFO:__main__:hit_charges n 202 std 0.0940629 [expect 0.1+-0.1] mean 0.968465 [expect 1.0+-0.1] 
    INFO:__main__:testTime
    INFO:__main__:hit_times n 497 std 1.18244 [expect 1.2+-0.1] mean 99.7021 


Repeatedly getting roughly twice the time hits 
compared to charge hits ? This must be due to the time offset ?


"""
import logging 
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)   # smth funny with chroma logging have to config here rather than in main

#from unittest_find import unittest
import unittest
import numpy as np

from chroma.geometry import Solid, Geometry, vacuum
from chroma.loader import create_geometry_from_obj
from chroma.detector import Detector
from chroma.make import box
from chroma.sim import Simulation
from chroma.event import Photons

from chroma.demo.optics import r7081hqe_photocathode


global sim
sim = None


class TC(unittest.TestCase):
    """
    # mis-usage, but deservedly so : nose is so much better
    """
    def runTest(self):
        pass


def _setup():
    # Setup geometry
    log.info("_setup")
    cube = Detector(vacuum)
    cube.add_pmt(Solid(box(10.0,10,10), vacuum, vacuum, surface=r7081hqe_photocathode))
    cube.set_time_dist_gaussian(1.2, -6.0, 6.0)
    cube.set_charge_dist_gaussian(1.0, 0.1, 0.5, 1.5)

    global sim
    geo = create_geometry_from_obj(cube, update_bvh_cache=False)
    sim = Simulation(geo, geant4_processes=0)
 

def _photons( nphotons, toffset=0. ):
    pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
    dir = np.tile([0,0,1], (nphotons,1)).astype(np.float32)
    pol = np.zeros_like(pos)
    phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
    pol[:,0] = np.cos(phi)
    pol[:,1] = np.sin(phi)
    t = np.zeros(nphotons, dtype=np.float32) + toffset 
    wavelengths = np.empty(nphotons, np.float32)
    wavelengths.fill(400.0)

    photons = Photons(pos=pos, dir=dir, pol=pol, t=t,
                      wavelengths=wavelengths)
    return photons


def _hit_times():
    # Run only one photon at a time
    log.info("testTime")

    photons = _photons( 1, 100.0 )  # Avoid negative photon times

    hit_times = []
    miss = 0 
    rolls = 1000

    # propagating a single photon, 1000 times
    for ev in sim.simulate(photons for i in xrange(rolls)):
        if ev.channels.hit[0]:
            hit_times.append(ev.channels.t[0])
        else:
            miss += 1

    hit_times = np.array(hit_times)

    assert len(hit_times) + miss == rolls 
    log.info("hit_times n %s miss %s std %s [expect 1.2+-0.1] mean %s " % (len(hit_times),miss, hit_times.std(), hit_times.mean()) )
    return hit_times



def _hit_charges():
    # Run only one photon at a time
    log.info("testCharge")
    hit_charges = []
    miss = 0 
    rolls = 1000
    photons = _photons(1)
    for ev in sim.simulate(photons for i in xrange(rolls)):
        if ev.channels.hit[0]:
            hit_charges.append(ev.channels.q[0])
        else:
            miss += 1 
    hit_charges = np.array(hit_charges)

    assert len(hit_charges) + miss == rolls 
    log.info("hit_charges n %s miss %s  std %s [expect 0.1+-0.1] mean %s [expect 1.0+-0.1] " % (len(hit_charges),miss, hit_charges.std(), hit_charges.mean()) )
    return hit_charges


class TestDetector(unittest.TestCase):
    def setUp(self):
        _setup(self)
       
    def testTime(self):
        '''Test PMT time distribution'''
        hit_times = _hit_times()
        self.assertAlmostEqual(hit_times.std(),  1.2, delta=1e-1)

    def testCharge(self):
        '''Test PMT charge distribution'''
        hit_charges = _hit_charges()
        self.assertAlmostEqual(hit_charges.mean(),  1.0, delta=1e-1)
        self.assertAlmostEqual(hit_charges.std(), 0.1, delta=1e-1)



def umain():
    unittest.main()
    #t =  TestDetector()
    #t.setUp()
    #t.testCharge()
    #t.testTime()

    #suite = unittest.TestLoader().loadTestsFromTestCase(TestDetector)
    #unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO)
    log.info("main")


    self = TC()   
    _setup()
    hit_charges = _hit_charges()
    #self.assertAlmostEqual(hit_charges.mean(),  1.0, delta=1e-1)
    #self.assertAlmostEqual(hit_charges.std(), 0.1, delta=1e-1)

    hit_times = _hit_times()
    #self.assertAlmostEqual(hit_times.std(),  1.2, delta=1e-1)
    


