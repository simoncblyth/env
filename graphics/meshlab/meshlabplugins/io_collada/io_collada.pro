#include ($$(MESHLAB_DIR)/shared.pri)

TEMPLATE      = lib
CONFIG       += plugin

GLEWDIR = $$(MESHLAB_DIR)/external/glew-1.7.0
INCLUDEPATH  *= $$(MESHLAB_DIR) $$(MESHLAB_VCGDIR) $$GLEWDIR/include

LIBS += $$(MESHLAB_DIR)/common/libcommon.dylib


DEFINES += "SCB_COLLADA_GEOMETRY_CACHE" 

QT += opengl
QT += xml 
QT += xmlpatterns
QT += script

HEADERS       += io_collada.h \
		$$(MESHLAB_VCGDIR)/wrap/io_trimesh/export_dae.h \
		$$(MESHLAB_VCGDIR)/wrap/io_trimesh/import_dae.h \
		$$(MESHLAB_VCGDIR)/wrap/dae/util_dae.h \
		$$(MESHLAB_VCGDIR)/wrap/dae/colladaformat.h \
		$$(MESHLAB_VCGDIR)/wrap/dae/xmldocumentmanaging.h


SOURCES       += io_collada.cpp \
        $$(MESHLAB_VCGDIR)/wrap/dae/xmldocumentmanaging.cpp

TARGET        = io_collada

QT           += xml opengl

