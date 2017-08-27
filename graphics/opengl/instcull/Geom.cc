#include <iostream>

#include "Buf.hh"
#include "BB.hh"
#include "Tra.hh"

#include "Primitives.hh"


#include "Geom.hh"


Geom::Geom(char shape_)
    :
    shape(shape_),
    num_vert(0),
    num_inst(0),
    num_viz(0),
    itra(NULL),
    ctra(NULL),
    prim(NULL),
    eidx(NULL),
    vbuf(NULL),  
    ebuf(NULL),  
    vbb(NULL),
    ibuf(NULL),  
    ibb(NULL),
    cbuf(NULL)
{
    init();
}

void Geom::init()
{
    if(shape == 'S')      initSpiral();
    else if(shape == 'G') initGlobe();
    else if(shape == 'L') initGlobeLOD();    
}

void Geom::setTransforms(Tra* tra)
{
    num_inst = tra->num_items() ; 
    ibuf = tra->buf ; 
    ibb = tra->bb ; 
    ce = ibb->get_center_extent();

    ctra = new Tra(num_inst, 'I') ;   // identity mat4 
    cbuf = ctra->buf ;
}

void Geom::setPrim(Prim* prim_)
{
    prim = prim_ ; 
    eidx = &prim->eidx ; 

    vbuf = prim->vbuf ; 
    ebuf = prim->ebuf ; 
    vbb = prim->bb ; 

    num_vert = vbuf->num_items ; 
}

void Geom::initSpiral()
{
    itra = new Tra(300, 'S') ;
    setTransforms(itra);

    Tri*  tri = new Tri(0.05f, 0.05f, 0.f,  0.f, 0.f, 0.f ); 
    setPrim((Prim*)tri);
}


void Geom::initGlobe()
{
    //Tri*  tri = new Tri(1.3333f, 1.f, 0.f,  0.f, 0.f, 0.f ); 
    //Prim* prim = (Prim*)tri ;

    //Cube* cube = new Cube(5.f, 5.f, 5.f,  0.f, 0.f, 0.f ); 
    //Prim* prim = (Prim*)cube ;

    Sphere* sphere = new Sphere(4u, 5.f); 
    Prim* prim = (Prim*)sphere ;

    setPrim(prim);

    unsigned num_polar = 300 ; 
    unsigned num_azimuth = 300 ; 
    Tra* tra = Tra::MakeGlobe(1000.f, num_azimuth, num_polar );

    setTransforms(tra);
}


void Geom::initGlobeLOD()
{
    Tri*  tri = new Tri(1.3333f, 1.f, 0.f,  0.f, 0.f, -2.f ); 
    Cube* cube = new Cube(1.f, 1.f, 1.f,  0.f, 0.f, 0.f ); 
    Sphere* sphere = new Sphere(4u, 1.5f, 0.f, 0.f, 0.f ); 

    std::vector<Prim*> prims ; 
    prims.push_back(tri);
    prims.push_back(cube);
    prims.push_back(sphere);

    Prim* prim = Prim::Concatenate(prims);
    setPrim(prim);

    unsigned num_polar = 200 ; 
    unsigned num_azimuth = 200 ; 
    Tra* tra = Tra::MakeGlobe(100.f, num_azimuth, num_polar );

    setTransforms(tra);
}


