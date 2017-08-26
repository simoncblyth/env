
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
#include "Shader.hh"

int main(int, char** argv)
{
    Tri*  tri = new Tri(1.3333f, 1.f, 0.f,  0.f, 0.f, -2.f ); 
    Cube* cube = new Cube(1.f, 1.f, 1.f,  0.f, 0.f, 0.f ); 
    Sphere* sphere = new Sphere(4u, 0.5f, 0.f, 0.f, 0.f ); 

    std::vector<Prim*> prims ; 
    prims.push_back(tri);
    prims.push_back(cube);
    prims.push_back(sphere);

    Prim* prim = Prim::Concatenate(prims);



    Frame frame(argv[0]) ; 
    Shader sh ; 

    bool wire = true ; 

    Buf* v = prim->vbuf ;
    Buf* e = prim->ebuf ;

    Comp comp ; 
    comp.setCenterExtent( prim->ce );
    comp.setEye( 0, 0,  1)  ;   // position eye along +z 
    comp.setLook(0, 0,  0)  ;   // center of region
    comp.setUp(  0, 1,  0)  ; 
    comp.setFocus( prim->ce.w, 10.f );  // near/far heuristic from extent of region of interest, near = extent/factor ; far = extent*factor

    comp.update();

    v->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    e->upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

    GLuint vao = sh.createVertexArray( v->id, e->id );

    glUseProgram( sh.draw->program );
    glBindVertexArray( vao );
    
    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, wire ? GL_LINE : GL_FILL );

    while (!glfwWindowShouldClose(frame.window))
    {
        glfwGetFramebufferSize(frame.window, &comp.cam->width, &comp.cam->height);
        glViewport(0, 0, comp.cam->width, comp.cam->height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        float t = (float)glfwGetTime(); 
        comp.setEye( 2*glm::cos(t), 0, 2*glm::sin(t) )  ; 
        comp.update();

        sh.updateMVP(comp.world2clip) ; 

        // gives same outcome when all elements are drawn together or separate
        //glDrawElements(GL_TRIANGLES, e->num_items, GL_UNSIGNED_INT, (void*)0 );

        for(unsigned lod=0 ; lod < prim->eidx.size() ; lod++)
        {
            const glm::uvec4& eidx = prim->eidx[lod] ;
            unsigned elem_offset = eidx.x ; 
            unsigned elem_count  = eidx.y ; 
            glDrawElements(GL_TRIANGLES, elem_count, GL_UNSIGNED_INT, (void*)(elem_offset*sizeof(unsigned)) );
        }
    
        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }
    sh.destroy();
    frame.destroy();
    return 0 ; 
}


