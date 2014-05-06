Chroma Server
==============

* https://github.com/mastbaum/chroma-server

This package lets you run a chroma server, which performs GPU-accelerated
photon propagation for clients. Clients send chroma-server a list of initial
photon vertices, and it replies with the final vertices of detected photons.

The chroma-server:

* https://github.com/mastbaum/chroma-server/blob/master/chroma_server/server.py

appears to now be integrated with chroma:

* https://bitbucket.org/chroma/chroma/src/b565b38ae23a5b7522b54af51091e2f7c4267c9c/bin/chroma-server

The original not using python objects for communication however, 
so it might be more directly relevant for usage from inside Geant4 C++.



Transport Mechanism : Feeding the server from Geant4 C++
-----------------------------------------------------------

#. Geant4 process collects `G4Track` into a `ChromaPhotonList` `TObject` 
   containing contains `std::vector<float>` and `std::vector<int>`

#. My simple class `ZMQRoot`, serializes the `ChromaPhotonList:TObject` into bytes
   (using ROOT TMessage) and thence into a ZMQ message thats sent over the wire to server

#. Sever deserializes back into `ChromaPhotonList:TObject` that then gets converted
   to numpy arrays which can be communicated to GPU using PyCUDA/Chroma capabilities


Alternatives to ROOT TObject Serialization ?
----------------------------------------------------

boost serialization
~~~~~~~~~~~~~~~~~~~~~

* http://www.boost.org/doc/libs/1_43_0/libs/serialization/doc/index.html

numpy from C ?
~~~~~~~~~~~~~~~~

pyzmq is able to send/receive numpy arrays (using multipart ZMQ message) with 
the dtype riding along as a json dict.

* http://zeromq.github.io/pyzmq/serialization.html
* https://github.com/lebedov/msgpack-numpy

Create numpy arrays in C directly from `std::vector<float>` ?
My mysql numpy did something similar from mysql query results.

* http://stackoverflow.com/questions/214549/how-to-create-a-numpy-record-array-from-c 
* http://stackoverflow.com/questions/18780570/passing-a-c-stdvector-to-numpy-array-in-python
* http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
* http://realgonegeek.blogspot.tw/2013/08/how-to-pass-c-array-to-python-solution.html

To do this conveniently would need to use cython

* https://gist.github.com/GaelVaroquaux/1249305









