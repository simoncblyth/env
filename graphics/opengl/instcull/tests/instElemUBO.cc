#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Prog.hh"
#include "Frame.hh"
#include "Buf.hh"

#include "Primitives.hh"
#include "Tra.hh"

#include "Geom.hh"
#include "Comp.hh"
#include "Cam.hh"

#include "SContext.hh"
#include "InstShader.hh"

int main(int argc, char** argv)
{
    Frame frame(argv[0],2880,1800) ; 

    SContext context ;
    InstShader is(&context) ; 

    bool wire = true ; 
    Geom* geom = new Geom('L');

    Comp comp ;
    comp.aim(geom->ce);
    comp.setUp( 0., 0., 1. ); 
    comp.setEye(   1, 0., 0. )  ; 
    comp.setLook( -1, 0., 0. )  ; 
    comp.update();
    comp.dump();

    geom->vbuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    geom->ebuf->upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    geom->ibuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    GLuint vao = is.createVertexArray(  geom->ibuf->id, geom->vbuf->id, geom->ebuf->id );

    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, wire ? GL_LINE : GL_FILL );

    glUseProgram(is.prog->program);
    glBindVertexArray( vao );

    int count(0) ; 

    while (!glfwWindowShouldClose(frame.window) && count++ < 3000 )
    {
        glfwGetFramebufferSize(frame.window, &comp.cam->width, &comp.cam->height);
        glViewport(0, 0, comp.cam->width, comp.cam->height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        float t = (float)glfwGetTime(); 
        //comp.setEye( glm::cos(t), glm::sin(t), 0 )  ; 
        comp.setEye(  1.f - t*0.1f, 0.f, 0.f )  ; 
        //float near = geom->ce.w*(0.5f + t*0.1f) ; 
        //comp.setNearFar( near, near*10. );
        comp.update();

        std::cout << " count " << count << " t " << t << std::endl ;
        context.update(comp.world2clip, comp.world2eye);

        //glDrawElementsInstanced(GL_TRIANGLES, geom->ebuf->num_items, GL_UNSIGNED_INT, NULL, geom->ibuf->num_items  ) ;

        for(unsigned lod=0 ; lod < geom->eidx->size() ; lod++)
        //unsigned lod=2 ;
        {
            const glm::uvec4& eidx = (*geom->eidx)[lod] ; 
            glDrawElementsInstanced(GL_TRIANGLES, eidx.y, GL_UNSIGNED_INT, (void*)(eidx.x*sizeof(unsigned)), geom->ibuf->num_items  ) ;
        }


        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    is.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


