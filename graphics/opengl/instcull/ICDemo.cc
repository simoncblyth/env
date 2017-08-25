#include <vector>
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

#include "CullShader.hh"
#include "InstShader.hh"


/*

NEXT:

* realistic MVP dependent frustum culling 
* lod streaming, starting with 2 lod levels : original and bbox

*/

ICDemo::ICDemo(const char* title) 
    :
    geom(new Geom('G')),
    comp(new Comp),
    frame(new Frame(title,2880,1800)),
    cull(new CullShader), 
    draw(new InstShader)
{
    init();
}

const unsigned ICDemo::QSIZE = 4*sizeof(float) ;

void ICDemo::init()
{
    geom->vbuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    geom->ebuf->upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    geom->ibuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    geom->cbuf->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);

    GU::errchk("ICDemo::init");

    cull->setupTransformFilter(geom->ibuf) ;

    comp->aim(geom->ce);
    comp->setEye( -0.01, 1, 1);  // avoid zeros, they tend to cause no-viz geometry 

    this->allVertexArray = draw->createVertexArray(geom->ibuf->id, geom->vbuf->id); 
    this->drawVertexArray = draw->createVertexArray(geom->cbuf->id, geom->vbuf->id ); 

    GU::errchk("ICDemo::init");
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
        // getting tunnel vision at the end of the glide ??
    }

    comp->update();
    draw->updateMVP(comp->world2clip);
}

void ICDemo::renderScene(float t)
{
    updateUniform(t);

    /////////// 1st pass : culling instance transforms 

    cull->applyTransformFilter(geom->cbuf) ;
 
    //bool use_cull = false ; 
    bool use_cull = true ; 

    std::cout 
          << " num_inst " << this->geom->num_inst 
          << " num_viz(GL_PRIMITIVES_GENERATED) " << cull->num_viz 
          << std::endl 
          ;
 
    /////////// 2nd pass : render just the surviving instances

    glUseProgram(draw->prog->program);
    glBindVertexArray( use_cull ? this->drawVertexArray : this->allVertexArray);

    unsigned num_draw = use_cull ? cull->num_viz : geom->num_inst ; 

    if(num_draw > 0)    
    {  
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geom->ebuf->id ); 
        glDrawElementsInstanced(GL_TRIANGLES, geom->ebuf->num_items, GL_UNSIGNED_INT, NULL, num_draw  ) ;
        //glDrawArraysInstanced( GL_TRIANGLES, 0, geom->num_vert,  num_draw );
     }   
}

void ICDemo::pullback()
{
    GLenum target = GL_TRANSFORM_FEEDBACK_BUFFER ;

    if(geom->num_viz == 0) return ; 
    unsigned viz_bytes = geom->num_viz*4*QSIZE ; 
    glGetBufferSubData( target , 0, viz_bytes, geom->ctra );
    geom->ctra->dump(geom->num_viz);    
}

void ICDemo::renderLoop()
{
    unsigned count(0); 
    glEnable(GL_DEPTH_TEST);
    
    while (!glfwWindowShouldClose(frame->window) && count++ < 2000 )
    {
        frame->listen();
        //std::cout << "ICDemo::renderLoop count " << count << std::endl ; 

        glfwGetFramebufferSize(frame->window, &comp->cam->width, &comp->cam->height);
        glViewport(0, 0, comp->cam->width, comp->cam->height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        renderScene((float)glfwGetTime());

        glfwSwapBuffers(frame->window);
    }
}

void ICDemo::destroy()
{
    cull->destroy();
    draw->destroy();
    frame->destroy();
}


