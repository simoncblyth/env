Serialization
===============

Current Solution : NPY
------------------------

Plumped for NPY (numpy ndarray native serialization) 


Alternatives serialization/deserialization to ROOT ?
-------------------------------------------------------

Ginormous monolithic ROOT dependency, just to be able to 
deserialize ChromaPhotonList objects.  

Requirements:

#. serialization usable from Geant4 C++
#. deserialization usable from Python
#. in future both usable from iOS
#. efficient handling of millions of photons!

ChromaPhotonList exists for transport alone, so 
can easily change the format to something else, 
eg go directly into capnproto structures, from 
Geant4 C++ 

Also consider what happens at other end, need to 
form numpy arrays with the data in order for 
easy uploading to GPU.

Non-requirements:

#. schema evolution, careful versioning


NEXT
~~~~

#. Make some comparisons for usecase of large numbers of floats, 
   no necessarily in `std::vector<float>`

contenders
~~~~~~~~~~~~~

* :google:`protobuf boost.serialize capnproto thrift`

#. capnproto, improvement on protobuf from former protobuf principal 
#. cereal
#. google protobuf
#. boost serialization http://www.ibm.com/developerworks/aix/library/au-boostserialization/
#. http://home.gna.org/autoserial/objects.html
#. http://thrift.apache.org

comparisons
~~~~~~~~~~~~

* https://www.igvita.com/2011/08/01/protocol-buffers-avro-thrift-messagepack/
* http://www.codeproject.com/Articles/225988/A-practical-guide-to-Cplusplus-serialization
* https://github.com/thekvs/cpp-serializers
* http://leopard.in.ua/2013/10/13/binary-serialization-formats/
* http://stackoverflow.com/questions/1061169/boost-serialization-vs-google-protocol-buffers
* https://github.com/mapbox/mapnik-vector-tile/issues/1
* http://en.wikipedia.org/wiki/Comparison_of_data_serialization_formats


capnproto
~~~~~~~~~~~

* Requires GCC 4.7+ or Clang 3.2+
* iOS ported: https://groups.google.com/forum/#!topic/capnproto/YIcDJzSQh34

* https://github.com/kentonv/capnproto
* http://kentonv.github.io/capnproto/
* https://trac.macports.org/browser/trunk/dports/devel/capnproto/Portfile
* http://blog.jparyani.com/pycapnp/  python binding

Random quote:

    It's extremely performant for C, because its format mimics the layout of C data
    structures in memory. So for C, the encoding time is zero - just dump the bytes
    as-is, pretty much.


Hmm, what about byte buffer reinterpretation as numpy array ? (a la MySQL-numpy).



cereal
~~~~~~~

* http://uscilab.github.io/cereal/index.html
* https://github.com/USCiLab/cereal


boost serialization
~~~~~~~~~~~~~~~~~~~~~~

* can monolithic-ness be avoided ? iOS ?

 



