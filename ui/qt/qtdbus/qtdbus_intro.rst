QtDbus Intro
=============

Follow along 

* http://developer.nokia.com/Community/Wiki/QtDbus_quick_tutorial

In short
-----------

::

    simon:qtdbus blyth$ ./qdbusxml2cpp.sh 
    qdbusxml2cpp -v -c DemoIf -p demoif.h:demoif.cpp com.nokia.Demo.xml
    Warning: deprecated annotation 'com.trolltech.QtDBus.QtTypeName.In1' found; suggest updating to 'org.qtproject.QtDBus.QtTypeName.In1'
    qdbusxml2cpp -v -c DemoIfAdaptor -a demoifadaptor.h:demoifadaptor.cpp com.nokia.Demo.xml
    Warning: deprecated annotation 'com.trolltech.QtDBus.QtTypeName.In1' found; suggest updating to 'org.qtproject.QtDBus.QtTypeName.In1'
    Warning: deprecated annotation 'com.trolltech.QtDBus.QtTypeName.In1' found; suggest updating to 'org.qtproject.QtDBus.QtTypeName.In1'
    Warning: deprecated annotation 'com.trolltech.QtDBus.QtTypeName.In1' found; suggest updating to 'org.qtproject.QtDBus.QtTypeName.In1'
    simon:qtdbus blyth$ 
    simon:qtdbus blyth$ qmake
    Project MESSAGE: qtAddLibrary: found framework QtDBus in directory /opt/local/Library/Frameworks
    Project MESSAGE: qtAddLibrary: found framework QtGui in directory /opt/local/Library/Frameworks
    Project MESSAGE: qtAddLibrary: found framework QtCore in directory /opt/local/Library/Frameworks
    simon:qtdbus blyth$ make
    ...
    simon:qtdbus blyth$ qtdbus.app/Contents/MacOS/qtdbus      ## dbus daemon needs to be running before starting the app
    MyDemo::MyDemo 


Interface description XML
--------------------------

.. literalinclude:: com.nokia.Demo.xml


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

.. literalinclude:: qdbusxml2cpp.sh


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

qdbus probing
----------------

::

    simon:~ blyth$ which dbus-send 
    /opt/local/bin/dbus-send

    simon:~ blyth$ which qdbus
    /opt/local/bin/qdbus


When dbus daemon is not running::

    simon:~ blyth$ qdbus com.nokia.Demo 
    Dynamic session lookup supported but failed: launchd did not provide a socket path, verify that org.freedesktop.dbus-session.plist is loaded!
    Could not connect to D-Bus server: org.freedesktop.DBus.Error.NoMemory: Not enough memory

When running::

    simon:~ blyth$ qdbus com.nokia.Demo 
    /

The daemon knows about the interface::

    simon:~ blyth$ qdbus com.nokia.Demo /
    signal void com.nokia.Demo.LateEvent(QString eventkind)
    method void com.nokia.Demo.SayBye()
    method void com.nokia.Demo.SayHello(QString name, QVariantMap customdata)
    method QDBusVariant org.freedesktop.DBus.Properties.Get(QString interface_name, QString property_name)
    method QVariantMap org.freedesktop.DBus.Properties.GetAll(QString interface_name)
    method void org.freedesktop.DBus.Properties.Set(QString interface_name, QString property_name, QDBusVariant value)
    method QString org.freedesktop.DBus.Introspectable.Introspect()
    method QString org.freedesktop.DBus.Peer.GetMachineId()
    method void org.freedesktop.DBus.Peer.Ping()


::

    simon:~ blyth$ dbus-send --type=method_call --print-reply \
    >     --dest=com.nokia.Demo / org.freedesktop.DBus.Introspectable.Introspect
    method return sender=:1.0 -> dest=:1.3 reply_serial=2
       string "<!DOCTYPE node PUBLIC "-//freedesktop//DTD D-BUS Object Introspection 1.0//EN"
    "http://www.freedesktop.org/standards/dbus/1.0/introspect.dtd">
    <node>
      <interface name="com.nokia.Demo">
        <method name="SayHello">
          <annotation value="QVariantMap" name="com.trolltech.QtDBus.QtTypeName.In1"/>
          <arg direction="in" type="s" name="name"/>
          <arg direction="in" type="a{sv}" name="customdata"/>
        </method>
        <method name="SayBye"/>
        <signal name="LateEvent">
          <arg direction="out" type="s" name="eventkind"/>
        </signal>
      </interface>
      <interface name="org.freedesktop.DBus.Properties">
        <method name="Get">
          <arg name="interface_name" type="s" direction="in"/>
          <arg name="property_name" type="s" direction="in"/>
          <arg name="value" type="v" direction="out"/>
        </method>
        <method name="Set">
          <arg name="interface_name" type="s" direction="in"/>
          <arg name="property_name" type="s" direction="in"/>
          <arg name="value" type="v" direction="in"/>
        </method>
        <method name="GetAll">
          <arg name="interface_name" type="s" direction="in"/>
          <arg name="values" type="a{sv}" direction="out"/>
          <annotation name="org.qtproject.QtDBus.QtTypeName.Out0" value="QVariantMap"/>
        </method>
      </interface>
      <interface name="org.freedesktop.DBus.Introspectable">
        <method name="Introspect">
          <arg name="xml_data" type="s" direction="out"/>
        </method>
      </interface>
      <interface name="org.freedesktop.DBus.Peer">
        <method name="Ping"/>
        <method name="GetMachineId">
          <arg name="machine_uuid" type="s" direction="out"/>
        </method>
      </interface>
    </node>
    "


Remote signalling::

    simon:~ blyth$ qdbus com.nokia.Demo / com.nokia.Demo.SayBye

Over at the app console::

    simon:qtdbus blyth$ qtdbus.app/Contents/MacOS/qtdbus 
    MyDemo::MyDemo 
    MyDemo::SayBye 

Similarly::

    simon:~ blyth$ dbus-send --type=method_call --print-reply \
    >     --dest=com.nokia.Demo / com.nokia.Demo.SayBye
    method return sender=:1.0 -> dest=:1.5 reply_serial=2


What about methods with parameters::

    simon:~ blyth$ qdbus com.nokia.Demo / com.nokia.Demo.SayHello simon
    Invalid number of parameters
    simon:~ blyth$ qdbus com.nokia.Demo / com.nokia.Demo.SayHello      
    Error: org.freedesktop.DBus.Error.UnknownMethod
    No such method 'SayHello' in interface 'com.nokia.Demo' at object path '/' (signature '')
    simon:~ blyth$ 
    simon:~ blyth$ qdbus com.nokia.Demo / com.nokia.Demo.SayHello simon b
    Sorry, can't pass arg of type 'QVariantMap'.


* http://forum.kde.org/viewtopic.php?f=17&t=85292

Seems cannot do that from commandline arguments with ``qdbus`` tool. So..


Add method that just takes string arg for simplicity
------------------------------------------------------

Succeeds to invoke methods with string arguments in the running qt app::

    simon:~ blyth$ qdbus com.nokia.Demo / com.nokia.Demo.SayHelloThere simon
    simon:~ blyth$ qdbus com.nokia.Demo / com.nokia.Demo.SayHelloThere "http://simon?c=2,3,4&other=yes&yo"

::

    simon:qtdbus blyth$ ./run.sh 
    MyDemo::MyDemo 
    MyDemo::SayHelloThere [ "simon" ] 
    MyDemo::SayHelloThere [ "http://simon?c=2,3,4&other=yes&yo" ] 


Python dbus
--------------

For passing dicts need to go beyond ``qdbus``::

    simon:~ blyth$ sudo port install -v dbus-python26 
    Password:
    --->  Computing dependencies for gettext


* http://dbus.freedesktop.org/doc/dbus-python/
* http://cgit.freedesktop.org/dbus/dbus-python/tree/examples/example-client.py

* http://dbus.freedesktop.org/doc/dbus-python/doc/tutorial.html


Too many dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:~ blyth$ port deps dbus-python26
    Full Name: dbus-python26 @0.83.2_0
    Build Dependencies:   pkgconfig
    Library Dependencies: dbus, dbus-glib, gettext, glib2, libiconv, python26, py26-gobject
    simon:~ blyth$ 

