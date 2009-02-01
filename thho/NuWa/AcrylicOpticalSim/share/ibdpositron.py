#!/usr/bin/env python

'''
usage example:

  nuwa.py -n 200 ibdpositron.py 

'''

from AcrylicOpticalSim.AcSim import IBDPositron
eplus = IBDPositron(histogram_filename = "IBDpositron.root", seed = "0", nevts = "200")

if '__main__' == __name__:
    print __doc__
