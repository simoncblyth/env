/*

NEXT:

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
    geom(new Geom('L')),
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
    // clod houses multiple buffers to grab the LOD forked instance transforms
    clod->x = geom->ibuf->cloneZero(); // CPU allocates and fills with zeros
    clod->y = geom->ibuf->cloneZero();
    clod->z = geom->ibuf->cloneZero();

    clod->x->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);  // GPU allocates only, no copying 
    clod->y->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);
    clod->z->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);

    //clod->devnull = new Buf(0,0,NULL);  // suspect zero-sized buffer is handled different, so use 1-byte buffer
    clod->devnull = new Buf(0,1,NULL);
    clod->devnull->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);  // zero sized buffer used with workaround


    cull->setupFork(geom->ibuf, clod) ;

    num_lod = clod->num_buf(); 

    for(int i=0 ; i < num_lod ; i++) 
        this->drawVertexArray[i] = draw->createVertexArray(clod->at(i)->id, geom->vbuf->id, geom->ebuf->id ); 

#else
    geom->cbuf->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);
    cull->setupTransformFilter(geom->ibuf) ;

    this->drawVertexArray[0] = draw->createVertexArray(geom->cbuf->id, geom->vbuf->id, geom->ebuf->id ); 
#endif
    this->allVertexArray = draw->createVertexArray(geom->ibuf->id, geom->vbuf->id, geom->ebuf->id); 


    comp->aim(geom->ce);

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
        comp->setEye(    3. - t*0.1, 0, 0 );
        comp->setLook(       -t*0.1, 0, 0 );
        // getting tunnel vision at the end of the glide ?? just near clip perhaps?
    }
    comp->update();
    context->update(comp->world2clip, comp->world2eye);
}

void ICDemo::renderScene()
{
    std::string status = getStatus();
    float t = frame->updateWindowTitle(status.c_str());
    //std::cout << status << std::endl ; 

    updateUniform(t);

#ifdef WITH_LOD
    cull->applyFork() ; 
    cull->applyForkStreamQueryWorkaround() ; 
    cull->dump("ICDemo::renderScene");
    //cull->pullback() ; 

    glUseProgram(draw->prog->program);

    for(unsigned lod=0 ; lod < num_lod ; lod++)
    {
        glBindVertexArray( use_cull ? this->drawVertexArray[lod] : this->allVertexArray);  

        unsigned num_draw = use_cull ? clod->at(lod)->query_count : geom->num_inst ; 
        if(num_draw == 0) continue ;
 
        const glm::uvec4& eidx = (*geom->eidx)[lod] ; 
        glDrawElementsInstanced(GL_TRIANGLES, eidx.y, GL_UNSIGNED_INT, (void*)(eidx.x*sizeof(unsigned)), num_draw  ) ;
    }

#else
    cull->applyTransformFilter(geom->cbuf) ; 

    unsigned num_draw = use_cull ? geom->cbuf->query_count : geom->num_inst ; 
    if(num_draw == 0) return ; 

    glUseProgram(draw->prog->program);
    glBindVertexArray( use_cull ? this->drawVertexArray[0] : this->allVertexArray);  

    glDrawElementsInstanced(GL_TRIANGLES, geom->ebuf->num_items, GL_UNSIGNED_INT, NULL, num_draw  ) ;       

#endif


}

std::string ICDemo::getStatus()
{
    std::stringstream ss ; 
    ss 
        << " num_inst " << geom->num_inst 
#ifdef WITH_LOD
        << " clod " << clod->desc()
#else
        << " geom.cbuf.query_count " << geom->cbuf->query_count
#endif
        ; 

    return ss.str();
}


void ICDemo::renderLoop()
{
    unsigned count(0); 
    bool wire = true ; 
    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, wire ? GL_LINE : GL_FILL );
    
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
/*
    GLenum target = GL_TRANSFORM_FEEDBACK_BUFFER ;
    if(cull->num_viz == 0) return ; 
    unsigned viz_bytes = cull->num_viz*4*QSIZE ; 
    glGetBufferSubData( target , 0, viz_bytes, geom->ctra );
    geom->ctra->dump(cull->num_viz);    
*/
}

void ICDemo::destroy()
{
    cull->destroy();
    draw->destroy();
    frame->destroy();
}

