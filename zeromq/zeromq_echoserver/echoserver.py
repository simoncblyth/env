#!/usr/bin/env python
"""
"""
import os, time, zmq, logging
import array
import numpy as np

from chroma.rootimport import ROOT
if ROOT.gSystem.Load(os.environ["CHROMAPHOTONLIST_LIB"]) < 0:ROOT.gSystem.Exit(10)
if ROOT.gSystem.Load(os.environ["ZMQROOT_LIB"]) < 0:ROOT.gSystem.Exit(10)

log = logging.getLogger(__name__)


def deserialize_tbufferfile(s, cls=ROOT.ChromaPhotonList.Class()):
    """
    From chromaserver

    rebuilds a ROOT object from a TBufferFile buffer, given such a buffer as
    a string, list, or iterable, and the class of the object-to-be
    (e.g. ROOT.TH1F.Class()).
    """
    b = array.array('c', s)
    buf = ROOT.TBufferFile(ROOT.TBuffer.kRead, len(b), b, False, 0)
    o = buf.ReadObject(cls)
    return o


def deserialize( s ):
    """
    TMessage knows the class that its carrying 

    #. declining ownership avoids error when tmsg goes out of scope: 
       "pointer being freed was not allocated"
    #. is it leaking though ?

    """
    b = array.array('c', s)
        
    tmsg = ROOT.MyTMessage( b, len(b) )
    print repr(tmsg)
    ROOT.SetOwnership(tmsg, False)  

    obj = tmsg.MyReadObject()
    log.info("obj %s " % repr(obj))
    return obj


def serialize_tbufferfile(o, cls=ROOT.ChromaPhotonList.Class()):
    """
    From chromaserver

    serializes ROOT object `o` via a ROOT.TBufferFile, returns a character
    array.array of `o`'s contents
    """
    buf = ROOT.TBufferFile(ROOT.TBuffer.kWrite)
    buf.Reset()
    buf.WriteObjectAny(o, cls)

    mv = memoryview(bytearray(buf.Length())).tobytes()
    a = array.array('c', mv) 

    return a


def serialize( obj ):
    tmsg = ROOT.MyTMessage( ROOT.MyTMessage.kMESS_OBJECT )
    tmsg.WriteObject(obj)
    bufLen = tmsg.Length() 

    mv = memoryview(bytearray(bufLen)).tobytes()
    a = array.array('c', mv) 

    tmsg.SerializeIntoArray(a)
    return a

 

cpl_atts = 'pmtid polx poly polz px py pz t wavelength x y z'.split()

def examine_obj(obj):
    print repr(obj)
    print obj.__class__

    atts = cpl_atts
    vecs = dict(map(lambda att:[att,getattr(obj,att)], atts ))
    sizs = map(lambda att:vecs[att].size(), atts)
    print sizs
    assert len(set(sizs)) == 1
    size = sizs[0]
    for i in range(size)[:10]:
        vals = map(lambda att:vecs[att][i], atts)
        d = dict(zip(atts,vals))
        print d

def create_random_obj(n=100):
     cpl = ROOT.ChromaPhotonList()
     for _ in range(n):
         cpl.AddPhoton( *np.random.random(11) )
     return cpl




if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(name)-20s:%(lineno)-3d %(message)s")
    print zmq.pyzmq_version()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    config = os.environ['ECHO_SERVER_CONFIG']
    socket.bind( config )

    print repr(context)
    print repr(socket)

    log.info("bound to %s hwm %s " % ( config, socket.hwm ))


    copy = False
    while True:
        msg = socket.recv(copy=copy)
        print repr(msg)
 
        if copy == False: assert type(msg) == zmq.backend.cython.message.Frame
        if copy == True: assert type(msg) == str
        log.info("recv message of length %s " % len(msg))

        obj = deserialize( msg.bytes )
        examine_obj( obj )

        cpl = create_random_obj()
        newmsg = serialize(cpl)

        time.sleep(1)
        log.info("now send")
        socket.send(newmsg)
    pass
pass

   




