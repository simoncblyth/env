# === func-gen- : python/py3/py3 fgp python/py3/py3.bash fgn py3 fgh python/py3 src base/func.bash
py3-source(){   echo ${BASH_SOURCE} ; }
py3-edir(){ echo $(dirname $(py3-source)) ; }
py3-ecd(){  cd $(py3-edir); }
py3-dir(){  echo $LOCAL_BASE/env/python/py3/py3 ; }
py3-cd(){   cd $(py3-dir); }
py3-vi(){   vi $(py3-source) ; }
py3-env(){  elocal- ; }
py3-usage(){ cat << EOU

Notes on py3 changes
=======================


commands module has gone
-------------------------

::
 
    try:
        from commands import getstatusoutput 
    except ImportError:
        from subprocess import getstatusoutput 
    pass 
           

file has gone
---------------

::

   find . -name '*.py' -exec grep -l file\( {} \;


2to3
-----

* http://python3porting.com/2to3.html

::

    epsilon:opticks blyth$ which 2to3
    /Users/blyth/miniconda3/bin/2to3



bytes/str/unicode
---------------------

* https://stackoverflow.com/questions/16476553/getting-the-same-unicode-string-length-in-both-python-2-and-3
* http://python3porting.com/problems.html#nicer-solutions

::

    import sys, codecs
    if sys.version_info.major > 2:
        u_ = lambda _:_                            # py3 strings are unicode already 
        b_ = lambda _:codecs.latin_1_encode(_)[0]  # from py3 unicode string to bytes
        d_ = lambda _:codecs.latin_1_decode(_)[0]  # from bytes to py3 unicode string
    else:
        u_ = lambda _:unicode(_, "utf-8")          # py2 strings are bytes
        b_ = lambda _:_ 
        d_ = lambda _:_ 
    pass


TypeError: Object of type int64 is not JSON serializable
----------------------------------------------------------

Works in py2 gives TypeError in py3::

   json.dump( {"a":1, "b":np.int64(42) }, open("/tmp/d.json","w"))   

Fix with custom encode::

   json.dump( {"a":1, "b":np.int64(42) }, open("/tmp/d.json","w"), cls=NPEncoder )   

See opticks.ana.base::

     class NPEncoder(json.JSONEncoder)


NameError: name 'unicode' is not defined
------------------------------------------

* https://stackoverflow.com/questions/19877306/nameerror-global-name-unicode-is-not-defined-in-python-3

Python 3 renamed the unicode type to str, the old str type has been replaced by bytes

* http://python3porting.com/problems.html

The biggest problem you may encounter relates to one of the most important
changes in Python 3; strings are now always Unicode. This will simplify any
application that needs to use Unicode, which is almost any application that is
to be used outside of English-speaking countries.

Of course, since strings are now always Unicode, we need another type for
binary data. Python 3 has two new binary types, bytes and bytearrays. The bytes
type is similar to the the string type, but instead of being a string of
characters, it’s a string of integers. Bytearrays are more like a list, but a
list that can only hold integers between 0 and 255. A bytearray is mutable and
used if you need to manipulate binary data. Because it’s a new type, although
it also exists in Python 2.6, I’m mostly going to ignore it in this book and
concentrate on other ways of handling binary data.


NameError: name 'reduce' is not defined
------------------------------------------

::

    from functools import reduce    ## works in py2 and py3



EOU
}
py3-get(){
   local dir=$(dirname $(py3-dir)) &&  mkdir -p $dir && cd $dir

}
