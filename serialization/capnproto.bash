# === func-gen- : serialization/capnproto fgp serialization/capnproto.bash fgn capnproto fgh serialization
capnproto-src(){      echo serialization/capnproto.bash ; }
capnproto-source(){   echo ${BASH_SOURCE:-$(env-home)/$(capnproto-src)} ; }
capnproto-vi(){       vi $(capnproto-source) ; }
capnproto-env(){      elocal- ; }
capnproto-usage(){ cat << EOU

CAPNPROTO
==========

* http://kentonv.github.io/capnproto/
* http://kentonv.github.io/capnproto/install.html


From FAQ
-----------

* http://kentonv.github.io/capnproto/faq.html

How do I resize a list?  
~~~~~~~~~~~~~~~~~~~~~~~~~

Unfortunately, you can’t. You have to know the size of
your list upfront, before you initialize any of the elements. This is an
annoying side effect of arena allocation, which is a fundamental part of Cap’n
Proto’s design: in order to avoid making a copy later, all of the pieces of the
message must be allocated in a tightly-packed segment of memory, with each new
piece being added to the end. 

...

The only solution is to temporarily place your data into some other data
structure (an std::vector, perhaps) until you know how many elements you have,
then allocate the list and copy. On the bright side, you probably aren’t losing
much performance this way – using vectors already involves making copies every
time the backing array grows. It’s just annoying to code.



EOU
}

capnproto-name(){ echo capnproto-c++-0.4.1 ;}
capnproto-dir(){ echo $(local-base)/env/serialization/capnproto/$(capnproto-name) ; }
capnproto-cd(){  cd $(capnproto-dir); }
capnproto-mate(){ mate $(capnproto-dir) ; }
capnproto-get(){
   local dir=$(dirname $(capnproto-dir)) &&  mkdir -p $dir && cd $dir

   local url=https://capnproto.org/$(capnproto-name).tar.gz
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -O $url 
   [ ! -f "$nam" ] && tar zxf $tgz

}

capnproto-build(){
   capnproto-cd
   #./configure
   #make -j6 check
   #sudo make install
}
