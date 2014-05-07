#!/usr/bin/env python

import os, logging, array, ctypes
log = logging.getLogger(__name__)

from env.root.import_ROOT import ROOT     # avoids sys.argv kidnap
if ROOT.gSystem.Load(os.environ["ZMQROOT_LIB"]) < 0:ROOT.gSystem.Exit(10)


def deserialize_tbufferfile(s):
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


def serialize_tbufferfile(o, cls ):
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


def deserialize( s ):
    """
    TMessage knows the class that its carrying 

    #. declining ownership avoids error when tmsg goes out of scope: 
       "pointer being freed was not allocated"
    #. is it leaking though ?

    """
    b = array.array('c', s)
        
    tmsg = ROOT.MyTMessage( b, len(b) )
    ROOT.SetOwnership(tmsg, False)  

    obj = tmsg.MyReadObject()
    #log.info("obj %s " % repr(obj))
    return obj

def serialize( obj ):
    """
    :param obj: any TObject subclass instance

    """
    tmsg = ROOT.MyTMessage( ROOT.MyTMessage.kMESS_OBJECT )
    tmsg.WriteObject(obj)
    msgLen = tmsg.Length() 

    # which of these does less copying ?  TODO:some big data tests to see
    #a = array.array('c', memoryview(bytearray(msgLen)).tobytes())
    #a = ctypes.create_string_buffer(msgLen) 
    a = (ctypes.c_char * msgLen)()

    tmsg.CopyIntoArray(a)
    return a



if __name__ == '__main__':
    pass

 
