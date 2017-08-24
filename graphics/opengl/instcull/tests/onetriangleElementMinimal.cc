
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
#include "Tri.hh"

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


Have observed that placing geometry precisely on near or far planes
usually results in clipping, need to use small delta to make visible. 

    //float cz = -c.getFar() + 1e-4f ;    // triangle at z = -far is clipped (ndc_z = +1) , need to add some delta to be visible    (small tri in center of screen)
    //float cz = -c.getNear() - 1e-4f ;   // triangle at z = -near is clipped (ndc_z = -1) , need to subtract some delta to be visible (fills screen)
    //float cz = -(c.getFar() + c.getNear())/2.f ; 



*/

int main()
{
    Frame frame ; 
    Prog draw(vertSrc, NULL, fragSrc ) ; 
    draw.compile();
    draw.create();
    draw.link();

    float cz = -1000.f ; 

    Tri tri(1.3333f, 1.f, 0.f,  0.f, 0.f, cz ); 
    Buf* a = tri.vbuf ;
    Buf* e = tri.ebuf ; 

    const glm::vec4& ce = tri.ce ; 

    Comp comp ; 
    comp.setCenterExtent( ce );

    Vue& vue = *comp.vue ; 
    Cam& cam = *comp.cam ; 

    vue.setEye( 0, 0,  1)  ;   // position eye along +z 
    vue.setLook(0, 0,  0)  ;   // center of region
    vue.setUp(  0, 1,  0)  ; 

    float factor = 10.f ; 
    cam.setFocus( ce.w, factor );  // near/far heuristic from extent of region of interest, near = extent/factor ; far = extent*factor

    comp.update();
    comp.dump();
    comp.dumpPoints(tri.vert);
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
    upload(e, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

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

        float angle = (float)glfwGetTime(); 
        vue.setEye( 2*glm::cos(angle), 0, 2*glm::sin(angle) )  ; 
        comp.update();

        //uniform.ModelViewProjection = glm::translate(glm::mat4(1.f), glm::vec3(-0.25f, -0.25f, 0.f) );
        uniform.ModelViewProjection = comp.world2clip ; 

        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Uniform), &uniform);
    
        //glDrawArrays(GL_TRIANGLES, 0, a->num_items);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, e->id);
        glDrawElements(GL_TRIANGLES, e->num_items, GL_UNSIGNED_INT, (void*)0 );

        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    draw.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


