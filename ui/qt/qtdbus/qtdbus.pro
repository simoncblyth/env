
# generate the 4 demoif sources with ./qdbusxml2cpp.sh 
# before running qmake to create the Makefile

TEMPLATE = app
TARGET = 
DEPENDPATH += .
INCLUDEPATH += .

QT += dbus


# Input
HEADERS += demoif.h demoifadaptor.h mydemo.h
SOURCES += demoif.cpp demoifadaptor.cpp main.cpp mydemo.cpp
