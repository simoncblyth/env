xml-vi(){ vi $BASH_SOURCE ; }
xml-usage(){ cat << EOU

XML Parsers
==============


irrXML (zlib)
----------------

* http://www.ambiera.com/irrxml/

irrXML is a simple and fast open source xml parser for C++. Why another xml
parser? The strenghts of irrXML are its speed and its simplicity. It ideally
fits into realtime projects which need to read xml data without overhead, like
games. irrXML was originally written as part of the Irrlicht Engine but after
it has become quite mature it now has become a separate project.







EOU
}

exist-(){   [ -r $ENV_HOME/xml/exist.bash ] && . $ENV_HOME/xml/exist.bash && exist-env $* ; }
modjk-(){   [ -r $ENV_HOME/xml/modjk.bash ] && . $ENV_HOME/xml/modjk.bash && modjk-env $* ; }

