#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Prog.hh"
#include "Frame.hh"
#include "Buf.hh"

#include "Primitives.hh"
#include "Tra.hh"

#include "Comp.hh"
#include "Cam.hh"

#include "InstRenderer.hh"

int main()
{
    Frame frame ; 
    InstRenderer ir ; 

    bool wire = true ; 
    float eyera = 3.f ; 

    //Tri*  tri = new Tri(1.3333f, 1.f, 0.f,  0.f, 0.f, -10.f ); 
    //Prim* prim = (Prim*)tri ;

    Cube* cube = new Cube(1.f, 1.f, 1.f,  0.f, 0.f, 0.f ); 
    Prim* prim = (Prim*)cube ;

    //Sphere* sphere = new Sphere(); 
    //Prim* prim = (Prim*)sphere ;

    Tra* tr = new Tra(10*10,'G' );

    Buf* i = tr->buf ;
    Buf* a = prim->vbuf ;
    Buf* e = prim->ebuf ; 

    Comp comp ;
    comp.aim(tr->ce);
    comp.update();

    comp.dump();
    comp.dumpPoints(prim->vert);
    comp.dumpFrustum();

    a->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    e->upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    i->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    GLuint vao = ir.createVertexArray(  i->id, a->id );

    glEnable(GL_DEPTH_TEST);
    glUseProgram(ir.draw->program);
    glBindVertexArray( vao );

    while (!glfwWindowShouldClose(frame.window))
    {
        glfwGetFramebufferSize(frame.window, &comp.cam->width, &comp.cam->height);
        glViewport(0, 0, comp.cam->width, comp.cam->height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        float angle = (float)glfwGetTime(); 
        comp.setEye( eyera*glm::cos(angle), 0, eyera*glm::sin(angle) )  ; 
        comp.update();

        ir.updateMVP(comp.world2clip);

        if(wire) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, e->id); 
        glDrawElementsInstanced(GL_TRIANGLES, e->num_items, GL_UNSIGNED_INT, NULL, i->num_items  ) ;

        if(wire) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 

        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    ir.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


