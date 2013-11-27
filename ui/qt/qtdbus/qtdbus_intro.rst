QtDbus Intro
=============

Follow along 

* http://developer.nokia.com/Community/Wiki/QtDbus_quick_tutorial


Interface description XML
--------------------------

.. include:: com.nokia.Demo.xml


Generate code from description with qtdbusxml2cpp
---------------------------------------------------

::

    simon:qtdbus blyth$ qdbusxml2cpp -h
    Usage: qdbusxml2cpp [options...] [xml-or-xml-file] [interfaces...]
    Produces the C++ code to implement the interfaces defined in the input file.

    Options:
      -a <filename>    Write the adaptor code to <filename>
      -c <classname>   Use <classname> as the class name for the generated classes
      -h               Show this information
      -i <filename>    Add #include to the output
      -l <classname>   When generating an adaptor, use <classname> as the parent class
      -m               Generate #include "filename.moc" statements in the .cpp files
      -N               Don't use namespaces
      -p <filename>    Write the proxy code to <filename>
      -v               Be verbose.
      -V               Show the program version and quit.

    If the file name given to the options -a and -p does not end in .cpp or .h, the
    program will automatically append the suffixes and produce both files.
    You can also use a colon (:) to separate the header name from the source file
    name, as in '-a filename_p.h:filename.cpp'.

    If you pass a dash (-) as the argument to either -p or -a, the output is written
    to the standard output


Generate Client Proxy Code from XML
--------------------------------------

::

    simon:qtdbus blyth$ pwd
    /Users/blyth/e/ui/qt/qtdbus
    simon:qtdbus blyth$ vi com.nokia.Demo.xml
    simon:qtdbus blyth$ which qdbusxml2cpp
    /opt/local/bin/qdbusxml2cpp
    simon:qtdbus blyth$ qdbusxml2cpp -v -c DemoIf -p demoif.h:demoif.cpp com.nokia.Demo.xml    
                        ##
                        ##  -v verbose
                        ##  -c DemoIf                    the classname to use
                        ##  -p demoif.h:demoif.cpp       write proxy code to these files
                        ##  
                        ##   generates demoif.{h,cpp}

    Warning: deprecated annotation 'com.trolltech.QtDBus.QtTypeName.In1' found; suggest updating to 'org.qtproject.QtDBus.QtTypeName.In1'
    simon:qtdbus blyth$ ll
    total 32
    drwxr-xr-x  3 blyth  staff   102 27 Nov 12:42 ..
    -rw-r--r--  1 blyth  staff    95 27 Nov 12:42 qtdbus_intro.rst
    -rw-r--r--  1 blyth  staff   430 27 Nov 12:43 com.nokia.Demo.xml
    -rw-r--r--  1 blyth  staff  1592 27 Nov 12:44 demoif.h
    -rw-r--r--  1 blyth  staff   666 27 Nov 12:44 demoif.cpp
    drwxr-xr-x  6 blyth  staff   204 27 Nov 12:44 .
    simon:qtdbus blyth$ 

* https://bugreports.qt-project.org/browse/QTBUG-14835

Attempt to follow the suggestion in the deprecatin warning, gives error and fails to generate, so live with the warning::

    perl -pi -e 's,trolltech,qtproject,g' com.nokia.Demo.xml 

    simon:qtdbus blyth$ qdbusxml2cpp -v -c DemoIf -p demoif.h:demoif.cpp com.nokia.Demo.xml
    Got unknown type `a{sv}'
    You should add <annotation name="org.qtproject.QtDBus.QtTypeName.In1" value="<type>"/> to the XML description

    simon:qtdbus blyth$ perl -pi -e 's,qtproject,trolltech,g' com.nokia.Demo.xml    ## back to original
    simon:qtdbus blyth$ qdbusxml2cpp -v -c DemoIf -p demoif.h:demoif.cpp com.nokia.Demo.xml
    Warning: deprecated annotation 'com.trolltech.QtDBus.QtTypeName.In1' found; suggest updating to 'org.qtproject.QtDBus.QtTypeName.In1'


* http://dbus.freedesktop.org/doc/dbus-specification.html


Generate Server Stub (Adapter code) from XML
-----------------------------------------------

Note ``-a`` rather than ``-p``::

    simon:qtdbus blyth$ qdbusxml2cpp -v -c DemoIfAdaptor -a demoifadaptor.h:demoifadaptor.cpp com.nokia.Demo.xml
    Warning: deprecated annotation 'com.trolltech.QtDBus.QtTypeName.In1' found; suggest updating to 'org.qtproject.QtDBus.QtTypeName.In1'
    Warning: deprecated annotation 'com.trolltech.QtDBus.QtTypeName.In1' found; suggest updating to 'org.qtproject.QtDBus.QtTypeName.In1'
    Warning: deprecated annotation 'com.trolltech.QtDBus.QtTypeName.In1' found; suggest updating to 'org.qtproject.QtDBus.QtTypeName.In1'
    simon:qtdbus blyth$ 


Put the above in a script
---------------------------

.. include:: qdbusxml2cpp.sh


Make a project 
----------------

::

    simon:qtdbus blyth$ qmake -project

Add to the qtdbus.pro generated::

    QT += dbus

Then:

#. `qmake` to generate a Makefile
#. `make` succeeds to moc the signal/slots and compile all the generated code by complains of lack of main at linking.

Run
----

::

    simon:qtdbus blyth$ qtdbus.app/Contents/MacOS/qtdbus 
    MyDemo::MyDemo 
    Dynamic session lookup supported but failed: launchd did not provide a socket path, verify that org.freedesktop.dbus-session.plist is loaded!


Try to query server
--------------------

::

    simon:~ blyth$ which qdbus
    /opt/local/bin/qdbus

    simon:~ blyth$ qdbus com.nokia.Demo 
    Dynamic session lookup supported but failed: launchd did not provide a socket path, verify that org.freedesktop.dbus-session.plist is loaded!
    Could not connect to D-Bus server: org.freedesktop.DBus.Error.NoMemory: Not enough memory


macports dbus
---------------

I was hoping could just use cmdline to server communication, but it seems this example
needs the dbus server.

* http://trac.macports.org/ticket/20645

::

    simon:~ blyth$ sudo port -v install dbus 







