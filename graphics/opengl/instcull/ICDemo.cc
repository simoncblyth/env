/*

NEXT:

* realistic MVP dependent frustum culling 
* lod streaming, starting with 2 lod levels : original and bbox

*/


#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "ICDemo.hh"
#include "Prog.hh"
#include "Frame.hh"
#include "Tra.hh"
#include "Buf.hh"
#include "Geom.hh"
#include "Comp.hh"
#include "Cam.hh"
#include "Vue.hh"
#include "BB.hh"
#include "G.hh"
#include "GU.hh"

#include "SContext.hh"

#ifdef WITH_LOD
#include "Buf4.hh"
#include "LODCullShader.hh"
#else
#include "CullShader.hh"
#endif

#include "InstShader.hh"



const unsigned ICDemo::QSIZE = 4*sizeof(float) ;

ICDemo::ICDemo(const char* title) 
    :
    geom(new Geom('G')),
    comp(new Comp),
    frame(new Frame(title,2880,1800)),
    context(new SContext),
#ifdef WITH_LOD
    cull(new LODCullShader(context)), 
    clod(new Buf4),
#else
    cull(new CullShader(context)), 
#endif
    draw(new InstShader(context)),
    use_cull(true)  
    //use_cull(false)  
{
    init();
}

void ICDemo::init()
{
    geom->vbuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    geom->ebuf->upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    geom->ibuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);

#ifdef WITH_LOD
    clod->x = geom->ibuf->cloneEmpty();
    clod->y = geom->ibuf->cloneEmpty();
    clod->x->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);
    clod->y->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);
    cull->setupFork(geom->ibuf, clod) ;
#else
    geom->cbuf->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);
    cull->setupTransformFilter(geom->ibuf) ;
#endif

    comp->aim(geom->ce);

    GU::errchk("ICDemo::init.0");

    this->allVertexArray = draw->createVertexArray(geom->ibuf->id, geom->vbuf->id); 
    this->drawVertexArray = draw->createVertexArray(geom->cbuf->id, geom->vbuf->id ); 

    GU::errchk("ICDemo::init.1");
}

void ICDemo::updateUniform(float t)
{
    if(geom->shape == 'S')
    {
        comp->setEye( glm::cos(t), 1, glm::sin(t) );
    }
    else
    {
        comp->setUp(     0, 0, 1 );
        comp->setEye(    1. - t*0.1, 0, 0 );
        comp->setLook(       -t*0.1, 0, 0 );
        // getting tunnel vision at the end of the glide ?? just near clip perhaps?
    }
    comp->update();
    context->updateMVP(comp->world2clip);
}

void ICDemo::renderScene()
{
    std::string status = getStatus();
    float t = frame->updateWindowTitle(status.c_str());
    //std::cout << status << std::endl ; 

    updateUniform(t);
    /////////// 1st pass : culling instance transforms 

#ifdef WITH_LOD
    cull->applyFork() ; 
#else
    cull->applyTransformFilter(geom->cbuf) ; 
#endif
    /////////// 2nd pass : render just the surviving instances

    unsigned num_draw = use_cull ? cull->num_viz : geom->num_inst ; 
    if(num_draw == 0) return ; 

    glUseProgram(draw->prog->program);
    glBindVertexArray( use_cull ? this->drawVertexArray : this->allVertexArray);  
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geom->ebuf->id ); 
    glDrawElementsInstanced(GL_TRIANGLES, geom->ebuf->num_items, GL_UNSIGNED_INT, NULL, num_draw  ) ;       

}

std::string ICDemo::getStatus()
{
    std::stringstream ss ; 
    ss 
        << " num_inst " << geom->num_inst 
        << " num_viz " << cull->num_viz
        ; 

    return ss.str();
}


void ICDemo::renderLoop()
{
    unsigned count(0); 
    glEnable(GL_DEPTH_TEST);
    
    while (!glfwWindowShouldClose(frame->window) && count++ < 2000 )
    {
        frame->listen();

        glfwGetFramebufferSize(frame->window, &comp->cam->width, &comp->cam->height);
        glViewport(0, 0, comp->cam->width, comp->cam->height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        renderScene();

        glfwSwapBuffers(frame->window);
    }
}

void ICDemo::pullback()
{
    GLenum target = GL_TRANSFORM_FEEDBACK_BUFFER ;
    if(cull->num_viz == 0) return ; 
    unsigned viz_bytes = cull->num_viz*4*QSIZE ; 
    glGetBufferSubData( target , 0, viz_bytes, geom->ctra );
    geom->ctra->dump(cull->num_viz);    
}

void ICDemo::destroy()
{
    cull->destroy();
    draw->destroy();
    frame->destroy();
}

