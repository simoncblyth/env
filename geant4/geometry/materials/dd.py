#!/usr/bin/env python
"""

scp://C//data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/materials/water.xml::

     12     <!-- This is taken from G4dyb's MaterialProperties.xml and massaged to fit -->
     13     <tabproperty name="WaterAbsorptionLength"
     14          type="ABSLENGTH"
     15          xunit="eV"
     16          yunit="cm"
     17          xaxis="PhotonEnergy"
     18          yaxis="AbsorptionLength">
     19 
     20          1.5498024  48.66809
     21          1.5694201  48.66475
     22          1.5734035  48.24148
     23          1.5774070  47.50742
     24          1.5814310  47.61985
     25          1.5854756  45.87167
     26          1.5895410  42.57084
     27          1.5936271  42.05114


    g4pb:materials blyth$ ./dd.py water.xml   
        ##
        ## converted 1st column wavelength eV to nm 
        ##           2nd column length     cm to mm   
        ## reversed order for ascending wavelength
        ##
    [[   196.00008271    273.2079    ]
     [   197.0000808     369.6279    ]
     [   198.00008346    491.5661    ]
     [   199.00008158    602.2004    ]
     [   200.00008175    691.5629    ]
     [   201.00008088    772.7537    ]
     [   202.0000887     831.7487    ]
     [   203.00009506    907.7467    ]
     ...
     [   778.00035328    420.5114    ]
     [   780.00029367    425.7084    ]
     [   782.00033277    458.7167    ]
     [   784.00034323    476.1985    ]
     [   786.00034538    475.0742    ]
     [   788.00031067    482.4148    ]
     [   790.00036179    486.6475    ]
     [   800.00033992    486.6809    ]]

    [[ 199.97458819    1.39      ]
     [ 299.76848327    1.3608    ]
     [ 309.80570885    1.36      ]
     [ 319.79428599    1.3595    ]
     [ 329.7453316     1.359     ]
     [ 339.77595144    1.3585    ]


"""
import os, sys
import lxml.etree as ET
import numpy as np
from common import as_optical_property_vector

parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()

if __name__ == '__main__':
   path = sys.argv[1]
   xml = parse_(path)
   for tp in xml.findall(".//tabproperty"):
       name = tp.attrib['name']
       type = tp.attrib['type']
       xunit = tp.attrib['xunit']
       yunit = tp.attrib['yunit']
       xaxis = tp.attrib['xaxis']
       yaxis = tp.attrib['yaxis']

       opv = as_optical_property_vector( tp.text, xunit=xunit, yunit=yunit )
       print opv


