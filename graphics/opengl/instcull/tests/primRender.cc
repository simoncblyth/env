
#include <vector>
#include <iostream>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "Frame.hh"
#include "Buf.hh"

#include "Primitives.hh"

#include "Comp.hh"
#include "Cam.hh"

#include "Prog.hh"
#include "Renderer.hh"

int main()
{
    Frame frame ; 
    Renderer rdr ; 


    bool wire = true ; 
    //-----
 
    //Tri*  tri = new Tri(1.3333f, 1.f, 0.f,  0.f, 0.f, -100.f ); 
    //Prim* prim = (Prim*)tri ; 

    Cube* cube = new Cube(1.f, 1.f, 1.f,  0.f, 0.f, 0.f ); 
    Prim* prim = (Prim*)cube ; 

    //Sphere* sphere = new Sphere(1.f, 0.f, 0.f, 0.f, 4 ); 
    //Prim* prim = (Prim*)sphere ; 

    //--------

    Buf* v = prim->vbuf ;
    Buf* e = prim->ebuf ;

    Comp comp ; 
    comp.setCenterExtent( prim->ce );
    comp.setEye( 0, 0,  1)  ;   // position eye along +z 
    comp.setLook(0, 0,  0)  ;   // center of region
    comp.setUp(  0, 1,  0)  ; 
    comp.setFocus( prim->ce.w, 10.f );  // near/far heuristic from extent of region of interest, near = extent/factor ; far = extent*factor

    comp.update();

    comp.dump();
    comp.dumpPoints(prim->vert);
    comp.dumpFrustum();

    v->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    e->upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

    GLuint vao = rdr.createVertexArray( v->id );

    glUseProgram( rdr.draw->program );
    glBindVertexArray( vao );
    
    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(frame.window))
    {
        glfwGetFramebufferSize(frame.window, &comp.cam->width, &comp.cam->height);
        glViewport(0, 0, comp.cam->width, comp.cam->height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        float angle = (float)glfwGetTime(); 
        comp.setEye( 2*glm::cos(angle), 0, 2*glm::sin(angle) )  ; 
        comp.update();

        rdr.updateMVP(comp.world2clip) ; 

        if(wire) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, e->id);
        glDrawElements(GL_TRIANGLES, e->num_items, GL_UNSIGNED_INT, (void*)0 );
        if(wire) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    rdr.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


