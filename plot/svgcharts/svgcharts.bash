# === func-gen- : plot/svgcharts fgp plot/svgcharts.bash fgn svgcharts fgh plot
svgcharts-src(){      echo plot/svgcharts/svgcharts.bash ; }
svgcharts-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svgcharts-src)} ; }
svgcharts-vi(){       vi $(svgcharts-source) ; }
svgcharts-env(){      elocal- ; }
svgcharts-usage(){ cat << EOU

SVGCHARTS
============

* https://pypi.python.org/pypi/svg.charts
* http://svg-charts.sourceforge.net/



G py26 install
--------------

::

    simon:svg.charts-2.0.9 blyth$ sudo python2.6 setup.py install
    copying build/lib/svg/__init__.py -> build/bdist.macosx-10.5-ppc/egg/svg
    creating build/bdist.macosx-10.5-ppc/egg/svg/charts
    copying build/lib/svg/charts/__init__.py -> build/bdist.macosx-10.5-ppc/egg/svg/charts
    ...
    creating /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/svg.charts-2.0.9-py2.6.egg
    Extracting svg.charts-2.0.9-py2.6.egg to /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages
    Adding svg.charts 2.0.9 to easy-install.pth file

    Installed /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/svg.charts-2.0.9-py2.6.egg
    Installed /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/cssutils-0.9.10-py2.6.egg
    Searching for python-dateutil==1.5
    Best match: python-dateutil 1.5
    Adding python-dateutil 1.5 to easy-install.pth file

    Using /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages
    Searching for lxml==2.3
    Best match: lxml 2.3
    Adding lxml 2.3 to easy-install.pth file

    Using /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages
    Finished processing dependencies for svg.charts==2.0.9
    simon:svg.charts-2.0.9 blyth$ 








EOU
}
svgcharts-dir(){ echo $(local-base)/env/plot/$(svgcharts-name) ; }
svgcharts-cd(){  cd $(svgcharts-dir); }
svgcharts-mate(){ mate $(svgcharts-dir) ; }
svgcharts-name(){ echo svg.charts-2.0.9 ; }
svgcharts-get(){
   local dir=$(dirname $(svgcharts-dir)) &&  mkdir -p $dir && cd $dir

   local nam=$(svgcharts-name)
   local zip=$nam.zip
   local url=https://pypi.python.org/packages/source/s/svg.charts/$zip

   [ ! -f "$zip" ] && curl -L -O "$url"
   [ ! -d "$nam" ] && unzip $zip

}
