// ggv --bnd

#include <cassert>


#include "NGLM.hpp"
#include "NPY.hpp"

#include "Opticks.hh"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBndLib.hh"


#include "PLOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    BRAP_LOG_ ;
    NPY_LOG_ ;
    GGEO_LOG_ ;

    LOG(info) << argv[0] ; 

    Opticks ok ;

    LOG(info) << " ok " ; 

    GBndLib* blib = GBndLib::load(&ok);

    LOG(info) << " loaded blib " ; 
    GMaterialLib* mlib = GMaterialLib::load(&ok);
    GSurfaceLib*  slib = GSurfaceLib::load(&ok);

    LOG(info) << " loaded all " 
              << " blib " << blib
              << " mlib " << mlib
              << " slib " << slib
              ;


    blib->setMaterialLib(mlib);
    blib->setSurfaceLib(slib);
    blib->dump();

    blib->save();             // only saves the guint4 bnd index
    blib->saveToCache();      // save float buffer too for comparison with wavelength.npy from GBoundaryLib with GBndLibTest.npy 
    LOG(info) << " after blib saveToCache " ; 
    blib->saveOpticalBuffer();
    LOG(info) << " after blib saveOpticalBuffer " ; 


/*
    const char* spec = "Vacuum/lvPmtHemiCathodeSensorSurface//Bialkali" ; // omat/osur/isur/imat
    assert(blib->contains(spec));
    bool flip = true ; 
    blib->add(spec, flip);
    blib->setBuffer(blib->createBuffer());
    blib->getBuffer()->save("/tmp/bbuf.npy");

*/

    return 0 ; 
}


/*

   running on empty dumps an npy file that causes a crash on subsequent run:

      PS C:\Users\ntuhep> rm  C:\tmp\TestIDPATH\GBndLib\GBndLibIndex.npy
 


*/

