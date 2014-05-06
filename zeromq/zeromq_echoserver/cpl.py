#!/usr/bin/env python
"""
"""
import os, logging
import array

from chroma.rootimport import ROOT
import numpy as np

if ROOT.gSystem.Load(os.environ["CHROMAPHOTONLIST_LIB"]) < 0:ROOT.gSystem.Exit(10)
if ROOT.gSystem.Load(os.environ["ZMQROOT_LIB"]) < 0:ROOT.gSystem.Exit(10)

log = logging.getLogger(__name__)


def serialize( obj ):
    tmsg = ROOT.MyTMessage( ROOT.MyTMessage.kMESS_OBJECT )
    tmsg.WriteObject(obj)
    bufLen = tmsg.Length() 

    mv = memoryview(bytearray(bufLen)).tobytes()
    a = array.array('c', mv) 

    tmsg.SerializeIntoArray(a)
    return a

 
def create_random_obj(n=100):
     cpl = ROOT.ChromaPhotonList()
     for _ in range(n):
         cpl.AddPhoton( *np.random.random(11) )
     return cpl




if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)-20s:%(lineno)-3d %(message)s")

    cpl = create_random_obj()
    print repr(cpl)
  
    msg = serialize(cpl)
    print repr(msg)

    






