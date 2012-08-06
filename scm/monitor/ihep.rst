
Backups of repos and tracs at IHEP
-------------------------------------

Doing this on dayabay.ihep.ac.cn ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Done:

#. python2.5.6 + sphinx + docutils etc... into  ~/local python
#. fabric + simplejson 
#. caution this will not work in the system python2.3 (used by apache/modpython/trac)
#. nginx running on 8080 (start nginx with command: ``nginx`` not ``nginx-start`` as do not have sudo and not needed for 8080 running)
#. add env symbolic link to nginx docs
#. hook up the javascript with link in _static


::

        g4pb-2:~ blyth$ ls -l ~/e/_static/
        total 8
        lrwxr-xr-x  1 blyth  staff  38 12 Jun 19:45 highstock -> /usr/local/env/plot/Highstock-1.1.6/js
        g4pb-2:~ blyth$ 


Hmm the link approach not working with nginx on WW

  * http://dayabay.phys.ntu.edu.tw/e/_static/highstock/highstock.js
  * http://dayabay.ihep.ac.cn:8080/e/_static/highstock/highstock.js




