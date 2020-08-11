# === func-gen- : doc/rst2pdf fgp doc/rst2pdf.bash fgn rst2pdf fgh doc
rst2pdf-src(){      echo doc/rst2pdf.bash ; }
rst2pdf-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rst2pdf-src)} ; }
rst2pdf-vi(){       vi $(rst2pdf-source) ; }
rst2pdf-env(){      elocal- ; }
rst2pdf-usage(){ cat << EOU

RST2PDF
==========

* https://code.google.com/p/rst2pdf/
* http://rst2pdf.ralsina.com.ar/
* http://rst2pdf.ralsina.com.ar/handbook.html
* http://ralsina.me/stories/BBS52.html


Aug 2020
----------

* https://rst2pdf.org/static/manual.pdf
* https://github.com/rst2pdf/rst2pdf
* https://akrabat.com/rst2pdf-back-from-the-dead/


Dependencies
--------------

* reportlab-

INSTALLS
----------

D
~~


Aug 2014, macports install::

    delta:~ blyth$ port info rst2pdf
    rst2pdf @0.93 (textproc, python)

    Description:          Create PDF from reStructuredText
    Homepage:             http://code.google.com/p/rst2pdf/

    Library Dependencies: python27, py27-reportlab, py27-docutils, py27-setuptools
    Runtime Dependencies: py27-pygments
    Platforms:            darwin
    License:              MIT
    Maintainers:          nomaintainer@macports.org
    delta:~ blyth$ 

This macport has an undeclared dependency,  py27-pdfrw



G
~~

After preparing reportlab with macports install::

    simon:rst2pdf-0.93 blyth$ sudo python setup.py install
    ...
    creating /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/rst2pdf-0.93.dev-py2.6.egg
    Extracting rst2pdf-0.93.dev-py2.6.egg to /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages
    Adding rst2pdf 0.93.dev to easy-install.pth file
    Installing rst2pdf script to /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin
    Installed /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/rst2pdf-0.93.dev-py2.6.egg
    ...

numpy ABI issue::

    simon:e blyth$ /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin/rst2pdf --help
    RuntimeError: module compiled against ABI version 2000000 but this version of numpy is 1000009
    Usage: rst2pdf [options]
    ...

    simon:e blyth$ python -c "import numpy ; print numpy.__version__ "
    1.6.2

USAGE
------

Readable PDF pages (not slide style though), including images on first try, despite warnings::

    simon:nov2013 blyth$ ./pdf.sh 
    RuntimeError: module compiled against ABI version 2000000 but this version of numpy is 1000009
    nov2013_gpu_nuwa.txt:6: (ERROR/3) Unknown interpreted text role "rawlink".

    .. role:: raw-link(rawlink)
       :format: html

    [WARNING] styles.py:548 Using undefined style 'green', aliased to style 'normal'.
    [WARNING] styles.py:548 Using undefined style 'red', aliased to style 'normal'.
    [WARNING] styles.py:548 Using undefined style 'blue', aliased to style 'normal'.
    [WARNING] styles.py:548 Using undefined style 'small', aliased to style 'normal'.
    simon:nov2013 blyth$ 


But to get it to match the s5 presentation would need considerable style tweaking ?

ALTERNATIVES
-------------

* http://code.google.com/p/wkhtmltopdf/




EOU
}

rst2pdf-dir(){ echo $(local-base)/env/doc/$(rst2pdf-name) ; }
#rst2pdf-name(){ echo rst2pdf-0.93 ; }
rst2pdf-name(){ echo rst2pdf ; }
rst2pdf-cd(){  cd $(rst2pdf-dir); }


rst2pdf-old-get(){
   local dir=$(dirname $(rst2pdf-dir)) &&  mkdir -p $dir && cd $dir

   local url=https://rst2pdf.googlecode.com/files/$(rst2pdf-name).tar.gz
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz
}


rst2pdf-get(){
   local dir=$(dirname $(rst2pdf-dir)) &&  mkdir -p $dir && cd $dir

   local nam=$(rst2pdf-name)

   [ ! -d $nam ] && git clone https://github.com/rst2pdf/rst2pdf $nam
   cd $nam
   
}


rst2pdf-slides-get(){
   local url="http://lateral.netmanagers.com.ar/static/rst2pdf-slides/slides.style"
   local nam=$(basename $url)
   [ ! -f "$nam" ] && curl -L -O $url
}

