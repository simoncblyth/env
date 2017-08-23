/*
Fairly Minimal Example of OpenGL Instancing 
===============================================

Using glVertexAttribDivisor, glDrawArraysInstanced
    
*/

#include <vector>
#include <iostream>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>


#include "Prog.hh"
#include "Frame.hh"
#include "Buf.hh"
#include "Pos.hh"

#include "Comp.hh"
#include "Vue.hh"
#include "Cam.hh"


const char* vertSrc = R"glsl(

    #version 400 core

    uniform MatrixBlock  
    {
        mat4 ModelViewProjection;
    } ;

    layout(location = 0) in vec4 vPosition ;
    void main()
    {
        gl_Position = ModelViewProjection * vPosition  ;  
    }
)glsl";


const char* fragSrc = R"glsl(
    #version 400 core 
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }
)glsl";


void upload(Buf* buf, GLenum target, GLenum usage )
{
    glGenBuffers(1, &buf->id);
    glBindBuffer(target, buf->id);
    glBufferData(target, buf->num_bytes, buf->ptr, usage);
    glBindBuffer(target, 0);
}



struct Uniform
{  
    glm::mat4 ModelViewProjection ;
};


/*

http://www.songho.ca/opengl/gl_projectionmatrix.html

Note that the frustum culling (clipping) is performed in the clip coordinates,
just before dividing by wc. The clip coordinates, xc, yc and zc are tested by
comparing with wc. If any clip coordinate is less than -wc, or greater than wc,
then the vertex will be discarded. 

Suspect my idea of clip space is off by factor of 2 ?


*/


int main()
{
    Frame frame ; 
    Prog draw(vertSrc, NULL, fragSrc ) ; 
    draw.compile();
    draw.create();
    draw.link();



    Comp comp ; 

    float factor = 10.f ; 
    float extent = 1.f ; 

    //comp.setCenterExtent( 100, 100, 100, 10 );
    comp.setCenterExtent(  0,  0,  -1,  extent );

    Vue& v = *comp.vue ; 
    Cam& c = *comp.cam ; 

    v.setEye( 0, 0,  1)  ;   // position eye along +z 
    v.setLook(0, 0,  0)  ;   // center of region
    v.setUp(  0, 1,  0)  ; 
    c.setFocus( extent, factor );  // near/far heuristic from extent of region of interest, near = extent/factor ; far = extent*factor


    comp.update();
    comp.dump();

    float x =  1.3333f ; 
    float y =  1.0f ; 
    float z = -1.0f ; 
    Buf* a = Pos::onetriangle(x, y, z);  // (-x,-y,z,1) (-x,+y,z,1) (x,0,z,1)

    comp.dumpTri(x,y,z);
    comp.dumpFrustum();



    GLuint uniformBlockIndex = glGetUniformBlockIndex(draw.program, "MatrixBlock") ;
    assert(uniformBlockIndex != GL_INVALID_INDEX);
    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(draw.program, uniformBlockIndex,  uniformBlockBinding );


    Uniform uniform ; 
    uniform.ModelViewProjection  = glm::mat4(1.f);

    GLuint ubo ; 
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(Uniform), &uniform, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint binding_point_index = 0 ;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point_index, ubo );


    GLuint vao ; 
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    upload(a, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    GLint vPosition = draw.getAttribLocation("vPosition");
    glBindBuffer(GL_ARRAY_BUFFER, a->id);
    glEnableVertexAttribArray(vPosition);
    glVertexAttribPointer(vPosition, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float),  NULL);
    

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(frame.window))
    {
        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );



        comp.update();

        //uniform.ModelViewProjection = glm::translate(glm::mat4(1.f), glm::vec3(-0.25f, -0.25f, 0.f) );
        uniform.ModelViewProjection = comp.world2clip ; 


        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Uniform), &uniform);
    

        glDrawArrays(GL_TRIANGLES, 0, a->num_items);

        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    draw.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


