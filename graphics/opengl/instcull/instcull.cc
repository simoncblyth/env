/*
Attempt Culling OpenGL Instances Via Geometry Shader and Transform Feedback 
=============================================================================

Aiming for a minimalist imp following technique of nature-

* http://rastergrid.com/blog/2010/02/instance-culling-using-geometry-shaders/

Multi-pass technique

1. cull against view frustum (or some arbitrary criteria in demo)
   using instance transforms (or simply offsets in demo) and extents of the instance
  
2. renders only those instances that are likely to be visible in the final scene

This can drastically reduce the amount of vertex data sent through the graphics pipeline.


Status
---------

* transform feedback succeeds to cull instances writing to the buffer
* BUT: subsequent render appears to not see the content ??

    
*/

#include <vector>
#include <iostream>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Prog.hh"
#include "Frame.hh"
#include "Buf.hh"
#include "Renderer.hh"


const char* vertCullSrc = R"glsl(

    #version 330 core
    in vec4 InstancePosition;
    out vec4 OrigPosition;
    flat out int objectVisible;
       
    void main() 
    {      
       OrigPosition = InstancePosition;
       objectVisible = InstancePosition.x > 0.5 ? 1 : 0;
    }

)glsl";

const char* geomCullSrc = R"glsl(

    #version 330 core
    layout(points) in; 
    layout(points, max_vertices = 1) out;

    in vec4 OrigPosition[1];
    flat in int objectVisible[1];
    out vec4 CulledPosition;

    void main() 
    {
       if ( objectVisible[0] == 1 ) 
       {   
          CulledPosition = OrigPosition[0];
          EmitVertex();
          EndPrimitive();
       }   
    }

)glsl";


const char* vertNormSrc = R"glsl(

    #version 330 core
    in vec4 VertexPosition;
    in vec4 VizInstancePosition;

    void main()
    {
        gl_Position = vec4( VertexPosition.x + VizInstancePosition.x, VertexPosition.y + VizInstancePosition.y, VertexPosition.z + VizInstancePosition.z, 1.0 ) ;  
    }


)glsl";

const char* fragNormSrc = R"glsl(

    #version 330 core 
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }

)glsl";





struct V { float x,y,z,w ; };
static const unsigned NUM_VPOS = 3 ; 

V vpos[NUM_VPOS] = 
{
    { -0.05f , -0.05f,  0.00f,  1.f }, 
    { -0.05f ,  0.05f,  0.00f,  1.f },
    {  0.00f ,  0.00f,  0.00f,  1.f }
};

static const unsigned NUM_IPOS = 8 ; 
V ipos[NUM_IPOS] = 
{
    {   0.5f ,  -0.5f,   0.f,  1.f },
    {   0.6f ,  -0.6f,   0.f,  1.f }, 
    {   0.3f ,  -0.3f,   0.f,  1.f },
    {   0.4f ,  -0.4f,   0.f,  1.f },
    {  -0.1f ,   0.1f,   0.f,  1.f }, 
    {  -0.2f ,   0.2f,   0.f,  1.f },
    {  -0.3f ,   0.3f,   0.f,  1.f },
    {  -0.4f ,   0.4f,   0.f,  1.f }
};



struct Att
{
   const Prog& prog ; 
   GLint       loc ; 
   Att(const Prog& prog_ , const char* name_ );
};

Att::Att(const Prog& prog_ , const char* name_ )
    :
    prog(prog_),
    loc(prog.getAttribLocation(name_))
{
    glEnableVertexAttribArray(loc);
    
    GLuint index = loc ; 
    GLint  size = 4 ;         // Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, 4.
    GLenum type = GL_FLOAT ;
    GLboolean normalized = GL_FALSE ;
    GLsizei stride = 4*sizeof(float) ;  // byte offset between consecutive generic vertex attributes
    const GLvoid* offset = NULL ;

    glVertexAttribPointer(index, size, type, normalized, stride, offset);
}




int main()
{
    Frame frame ; 

    Prog cull(vertCullSrc, geomCullSrc, NULL ) ; 
    cull.compile();
    cull.create();
    const char *vars[] = { "CulledPosition" };
    //glTransformFeedbackVaryings(cull.program, 1, vars, GL_SEPARATE_ATTRIBS);  
    glTransformFeedbackVaryings(cull.program, 1, vars, GL_INTERLEAVED_ATTRIBS); 
    cull.link();

    Prog norm(vertNormSrc, NULL, fragNormSrc ) ; 
    norm.compile();
    norm.create();
    norm.link();

    GLfloat feedback[NUM_IPOS*4];

    Buf v( sizeof(vpos),vpos ) ; 
    Buf i( sizeof(ipos),ipos ) ; 
    Buf c( sizeof(ipos),NULL ) ;   // CulledPosition subset of InstancePosition are copied in here

    Renderer rdr ; 
    rdr.upload(&v, GL_ARRAY_BUFFER             , GL_STATIC_DRAW);
    rdr.upload(&i, GL_ARRAY_BUFFER             , GL_STATIC_DRAW);
    rdr.upload(&c, GL_TRANSFORM_FEEDBACK_BUFFER, GL_STATIC_READ);


    ///////////////////////// PASS 1 : CULLING /////////////

    glUseProgram(cull.program);

    // culling pass needs access to instance positions in non-instanced manner
    glBindBuffer(GL_ARRAY_BUFFER, i.id);
    Att ci(cull, "InstancePosition");

    GLuint query;
    glGenQueries(1, &query);

    glEnable(GL_RASTERIZER_DISCARD);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, c.id );
    glBeginTransformFeedback(GL_POINTS);
    {
        glBeginQuery(GL_PRIMITIVES_GENERATED, query );
        {
            glDrawArrays(GL_POINTS, 0, NUM_IPOS );
        }
        glEndQuery(GL_PRIMITIVES_GENERATED); 
    }
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);

    glFlush();
    GLuint primitives;
    glGetQueryObjectuiv(query, GL_QUERY_RESULT, &primitives);
    std::cout << " GL_PRIMITIVES_GENERATED: " << primitives << std::endl ; 

   // Fetch and print results
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(feedback), feedback);
    for (int i = 0; i < NUM_IPOS; i++) 
    {
        printf("%f %f %f %f \n", feedback[i*4+0], feedback[i*4+1], feedback[i*4+2], feedback[i*4+3] );
    }

    ///////////////////////// PASS 2 : RENDER //////////////////////////

    glUseProgram(norm.program);
    glBindVertexArray(rdr.vao);

    glBindBuffer(GL_ARRAY_BUFFER, v.id);
    Att vp(norm, "VertexPosition"); 

    bool use_cull = true ; 
    //bool use_cull = false ; 
    if(use_cull)
    {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, c.id);     
    }
    else
    { 
        primitives = NUM_IPOS ; 
        glBindBuffer(GL_ARRAY_BUFFER, i.id);
    }
 
    Att ip(norm, "VizInstancePosition"); 
    glVertexAttribDivisor(ip.loc, 1 );
    

    glEnableVertexAttribArray(ip.loc);
    glEnableVertexAttribArray(vp.loc);
  

    while (!glfwWindowShouldClose(frame.window)  && primitives > 0)
    {
        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        {
            GLenum mode = GL_TRIANGLES ;
            GLint first = 0 ; 
            GLsizei count = NUM_VPOS ;  // number of indices to render, NB not number of prim
            GLsizei primCount = primitives  ;  
            glDrawArraysInstanced(mode, first, count, primCount );
            //glDrawArrays(mode, first, count );
        }


        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }


    rdr.destroy();
    cull.destroy();
    norm.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


