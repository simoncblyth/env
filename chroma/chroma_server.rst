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





