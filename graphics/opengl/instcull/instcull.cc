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
* subsequent render now sees just the selected instances transforms


What Not Working 
--------------------

Trying to render from the GL_TRANSFORM_FEEDBACK_BUFFER
with something like::

   glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, cbuf.id);    
 
Results in the render not seeing the instance postitions, 
gets all zeros resulting in all instances on top of
each other.

    
*/

#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Prog.hh"
#include "Frame.hh"
#include "Buf.hh"
#include "Renderer.hh"
#include "Att.hh"
#include "Transforms.hh"



//const unsigned LOC_InstancePosition = 0 ;  
const unsigned LOC_InstanceTransform = 0 ;  

const char* vertCullSrc = R"glsl(

    #version 400
    //layout( location = 0) in vec4 InstancePosition;
    layout( location = 0) in mat4 InstanceTransform ;
    out int objectVisible;
       
    void main() 
    {      
        //gl_Position = InstancePosition;
        vec4 tlate = InstanceTransform[3] ;
        gl_Position = tlate ; 
        objectVisible = tlate.x >= 0.f ? 1 : 0;
    }

)glsl";

const char* geomCullSrc = R"glsl(

    #version 400
    layout(points) in; 
    layout(points, max_vertices = 1) out;

    in int objectVisible[1] ;

    out vec4 CulledPosition;

    void main() 
    {
        if ( objectVisible[0] == 1 ) 
        {   
            CulledPosition = gl_in[0].gl_Position ;
            EmitVertex();
            EndPrimitive();
        }   
    }

)glsl";




const unsigned LOC_VertexPosition = 0 ;  
const unsigned LOC_VizInstancePosition = 1 ;  

const char* vertNormSrc = R"glsl(

    #version 400 core
    layout (location = 0) in vec4 VertexPosition;
    layout (location = 1) in vec4 VizInstancePosition;

    void main()
    {
        gl_Position = vec4( VertexPosition.x + VizInstancePosition.x, VertexPosition.y + VizInstancePosition.y, VertexPosition.z + VizInstancePosition.z, 1.0 ) ;  
    }

)glsl";

const char* fragNormSrc = R"glsl(

    #version 400 core 
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
    {  0.05f ,   0.05f,  0.00f,  1.f }, 
    {  0.05f ,  -0.05f,  0.00f,  1.f },
    {  0.00f ,   0.00f,  0.00f,  1.f }
};

static const unsigned NUM_IPOS = 8 ; 
V ipos_fix[NUM_IPOS] = 
{
    {   0.5f ,  -0.5f,   0.1f,  1.f },
    {  -0.1f ,   0.1f,   0.1f,  1.f }, 
    {   0.6f ,  -0.6f,   0.f,   1.f }, 
    {  -0.2f ,   0.2f,   0.f,   1.f },
    {   0.3f ,  -0.3f,   0.1f,  1.f },
    {  -0.3f ,   0.3f,   0.1f,  1.f },
    {   0.4f ,  -0.4f,   0.f,   1.f },
    {  -0.4f ,   0.4f,   0.f,   1.f }
};





int main()
{
    Frame frame ; 
       
    Q qq0("GL_MAX_TEXTURE_BUFFER_SIZE(texels)", GL_MAX_TEXTURE_BUFFER_SIZE);
    Q qq1("GL_MAX_UNIFORM_BLOCK_SIZE", GL_MAX_UNIFORM_BLOCK_SIZE);


    Prog cull(vertCullSrc, geomCullSrc, NULL ) ; 
    cull.compile();
    cull.create();

    // spacing the output from cull phase to write last vec4 of mat4
    const char *vars[] = { "gl_SkipComponents4", "gl_SkipComponents4", "gl_SkipComponents4",  "CulledPosition" };
    glTransformFeedbackVaryings(cull.program, 4, vars, GL_INTERLEAVED_ATTRIBS); 
    cull.link();

    Prog norm(vertNormSrc, NULL, fragNormSrc ) ; 
    norm.compile();
    norm.create();
    norm.link();

    unsigned num_inst = 10 ; 
    Transforms tr(num_inst, 4, 4, NULL) ; 
    tr.dump(); 

    Buf ibuf( tr.num_bytes(), tr.itra ) ; 
    Buf cbuf( tr.num_bytes(), NULL ) ;   // CulledPosition subset of InstancePosition are copied in here


    Buf vbuf( sizeof(vpos),vpos ) ; 

    Renderer rdr ; 
    rdr.upload( &ibuf , GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    rdr.upload( &cbuf , GL_ARRAY_BUFFER, GL_STREAM_DRAW);


    ///////////////////////// PASS 1 : CULLING /////////////

    glUseProgram(cull.program);

    // culling pass needs access to instance positions in non-instanced manner
    glBindBuffer(GL_ARRAY_BUFFER, ibuf.id);


    unsigned QSIZE = 4*sizeof(float) ;

    Att ci0(cull, LOC_InstanceTransform + 0 , 4, 4*QSIZE, 0 ); 
    Att ci1(cull, LOC_InstanceTransform + 1 , 4, 4*QSIZE, 1*QSIZE ); 
    Att ci2(cull, LOC_InstanceTransform + 2 , 4, 4*QSIZE, 2*QSIZE ); 
    Att ci3(cull, LOC_InstanceTransform + 3 , 4, 4*QSIZE, 3*QSIZE ); 

    glEnableVertexAttribArray(ci0.loc);
    glEnableVertexAttribArray(ci1.loc);
    glEnableVertexAttribArray(ci2.loc);
    glEnableVertexAttribArray(ci3.loc);


    GLuint query;
    glGenQueries(1, &query);

    glEnable(GL_RASTERIZER_DISCARD);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, cbuf.id );
    glBeginTransformFeedback(GL_POINTS);
    {
        glBeginQuery(GL_PRIMITIVES_GENERATED, query );
        {
            glDrawArrays(GL_POINTS, 0, tr.num_items() );
        }
        glEndQuery(GL_PRIMITIVES_GENERATED); 
    }
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);

    glFlush();
    GLuint nviz ;
    glGetQueryObjectuiv(query, GL_QUERY_RESULT, &nviz);
    std::cout << " GL_PRIMITIVES_GENERATED: " << nviz << std::endl ; 

   // Fetch and print results


    float* viz_ = new float[nviz*4*4] ;
    Transforms viz(nviz, 4, 4, viz_ );    
    assert( viz.num_bytes() == sizeof(float)*4*4*nviz ); 

    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, viz.num_bytes(), viz_ );

    viz.dump();    


   
    glUseProgram(0);
    ///////////////////////// PASS 2 : RENDER //////////////////////////


    rdr.upload( &vbuf , GL_ARRAY_BUFFER             , GL_STATIC_DRAW);

    while (!glfwWindowShouldClose(frame.window)  && nviz > 0)
    {
        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        {
            glUseProgram(norm.program);
            glBindVertexArray(rdr.vao);

            glBindBuffer(GL_ARRAY_BUFFER, vbuf.id);
            Att vp(norm, LOC_VertexPosition, 4, QSIZE, 0); 
            glEnableVertexAttribArray(vp.loc);
 
            bool use_cull = true ; 
            //bool use_cull = false ; 
              
            glBindBuffer(GL_ARRAY_BUFFER, use_cull ? cbuf.id : ibuf.id);         
            Att vip(norm, LOC_VizInstancePosition, 4, 4*QSIZE, 3*QSIZE ); 

            glEnableVertexAttribArray(vip.loc);  

            GLuint divisor = 1 ;   // number of instances between updates of ip.loc attribute , >1 will land that many instances on top of each other
            glVertexAttribDivisor(vip.loc, divisor );
            
            glEnableVertexAttribArray(vip.loc);

            glDrawArraysInstanced( GL_TRIANGLES, 0, NUM_VPOS, use_cull ? nviz : num_inst  );
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


