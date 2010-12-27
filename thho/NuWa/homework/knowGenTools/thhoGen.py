#!/usr/bin/env python

'''

An example job option script running GenHists algorithm.
It can be run like:

 nuwa.py -n XXX -o output.root thhoGen.py

The geometry, GenTools are also set up.

'''

import thhoGenConf 
she = thhoGenConf.Configure(do_detsim = False);

