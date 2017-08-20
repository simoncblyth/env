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


const unsigned LOC_InstanceTransform = 0 ;  

const char* vertCullSrc = R"glsl(

    #version 400

    layout( location = 0) in mat4 InstanceTransform ;
    out mat4 ITransform ;       

    void main() 
    {      
        ITransform = InstanceTransform ; 
    }
    // pass transform thru into geometry shader

)glsl";

const char* geomCullSrc = R"glsl(

    #version 400

    layout(points) in; 
    layout(points, max_vertices = 1) out;

    in mat4 ITransform[1] ;

    out vec4 VizTransform0 ;
    out vec4 VizTransform1 ;
    out vec4 VizTransform2 ;
    out vec4 VizTransform3 ;

    void main() 
    {
        if ( ITransform[0][3].x >= 0.f )   // arbitrary visibility criteria
        {    
            VizTransform0 = ITransform[0][0]  ;
            VizTransform1 = ITransform[0][1]  ;
            VizTransform2 = ITransform[0][2]  ;
            VizTransform3 = ITransform[0][3]  ;

            EmitVertex();
            EndPrimitive();
        }   
    }

)glsl";




const unsigned LOC_VertexPosition = 0 ;  
const unsigned LOC_VizInstanceTransform = 1 ;  

const char* vertNormSrc = R"glsl(

    #version 400 core
    layout (location = 0) in vec4 VertexPosition;
    layout (location = 1) in mat4 VizInstanceTransform ;
    void main()
    {
        gl_Position = VizInstanceTransform * VertexPosition ;
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



int main()
{
    Frame frame ; 
       
    Q qq0("GL_MAX_TEXTURE_BUFFER_SIZE(texels)", GL_MAX_TEXTURE_BUFFER_SIZE);
    Q qq1("GL_MAX_UNIFORM_BLOCK_SIZE", GL_MAX_UNIFORM_BLOCK_SIZE);


    Prog cull(vertCullSrc, geomCullSrc, NULL ) ; 
    cull.compile();
    cull.create();

    // spacing the output from cull phase to write last vec4 of mat4
    //const char *vars[] = { "gl_SkipComponents4", "gl_SkipComponents4", "gl_SkipComponents4",  "CulledPosition" };
    const char *vars[] = { "VizTransform0", "VizTransform1", "VizTransform2",  "VizTransform3" };
    glTransformFeedbackVaryings(cull.program, 4, vars, GL_INTERLEAVED_ATTRIBS); 
    cull.link();

    Prog norm(vertNormSrc, NULL, fragNormSrc ) ; 
    norm.compile();
    norm.create();
    norm.link();

    unsigned num_inst = 200 ; 
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
 
            //bool use_cull = true ; 
            bool use_cull = false ; 
              
            glBindBuffer(GL_ARRAY_BUFFER, use_cull ? cbuf.id : ibuf.id);    
     
            Att vip0(norm, LOC_VizInstanceTransform + 0, 4, 4*QSIZE, 0*QSIZE ); 
            Att vip1(norm, LOC_VizInstanceTransform + 1, 4, 4*QSIZE, 1*QSIZE ); 
            Att vip2(norm, LOC_VizInstanceTransform + 2, 4, 4*QSIZE, 2*QSIZE ); 
            Att vip3(norm, LOC_VizInstanceTransform + 3, 4, 4*QSIZE, 3*QSIZE ); 

            GLuint divisor = 1 ;   // number of instances between updates of ip.loc attribute , >1 will land that many instances on top of each other
            glVertexAttribDivisor(vip0.loc, divisor );
            glVertexAttribDivisor(vip1.loc, divisor );
            glVertexAttribDivisor(vip2.loc, divisor );
            glVertexAttribDivisor(vip3.loc, divisor );
            
            glEnableVertexAttribArray(vip0.loc);  
            glEnableVertexAttribArray(vip1.loc);  
            glEnableVertexAttribArray(vip2.loc);  
            glEnableVertexAttribArray(vip3.loc);  

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


