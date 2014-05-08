#
#
# Create Makefile with:
#     qmake
#
# Build with:
#     make
#
# Run with:
#     /tmp/qurl.app/Contents/MacOS/qurl
#
# Clean working copy:
#      make distclean
#



TEMPLATE = app
TARGET = 
DEPENDPATH += .
INCLUDEPATH += .

DESTDIR=/tmp


QT -= gui


# Input
SOURCES += qurl.cpp
