# === func-gen- : doc/reportlab fgp doc/reportlab.bash fgn reportlab fgh doc
reportlab-src(){      echo doc/reportlab.bash ; }
reportlab-source(){   echo ${BASH_SOURCE:-$(env-home)/$(reportlab-src)} ; }
reportlab-vi(){       vi $(reportlab-source) ; }
reportlab-env(){      elocal- ; }
reportlab-usage(){ cat << EOU

REPORTLAB
==========

* http://www.reportlab.com/
* http://www.reportlab.com/software/installation/

INSTALLS
---------


D
~~

Aug 2014, py27-reportlab from macports::

    delta:~ blyth$ port contents py27-reportlab
    Port py27-reportlab contains:
      /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/reportlab-3.1.8-py2.7.egg-info
      /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/reportlab/__init__.py
      ...
      /opt/local/share/doc/py27-reportlab/README.txt

    delta:~ blyth$ python -c "import reportlab"
    delta:~ blyth$ 


G
~~

macports, into py26::

    simon:nov2013 blyth$ port info py26-reportlab
    py26-reportlab @2.7 (python, textproc)
    Variants:             universal

    Description:          ReportLab is a software library that lets you directly create documents in Adobe's Portable Document Format (PDF) using the python programming language.
    Homepage:             http://www.reportlab.com/software/opensource/rl-toolkit/

    Library Dependencies: python26, py26-pil
    Platforms:            darwin, freebsd
    License:              BSD
    Maintainers:          stromnov@macports.org, openmaintainer@macports.org

    sudo port -v install py26-reportlab


EOU
}
reportlab-dir(){ echo $(local-base)/env/doc/doc-reportlab ; }
reportlab-cd(){  cd $(reportlab-dir); }
reportlab-mate(){ mate $(reportlab-dir) ; }
reportlab-get(){
   local dir=$(dirname $(reportlab-dir)) &&  mkdir -p $dir && cd $dir

}
