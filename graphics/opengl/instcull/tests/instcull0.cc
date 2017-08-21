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


Culling gives expected on 1st call, not on subsequent

::

     num_tr 200 num_viz(GL_PRIMITIVES_GENERATED) 100 viz_bytes 6400
     num_tr 200 num_viz(GL_PRIMITIVES_GENERATED) 200 viz_bytes 12800
     num_tr 200 num_viz(GL_PRIMITIVES_GENERATED) 200 viz_bytes 12800
     num_tr 200 num_viz(GL_PRIMITIVES_GENERATED) 200 viz_bytes 12800
     num_tr 200 num_viz(GL_PRIMITIVES_GENERATED) 200 viz_bytes 12800
     num_tr 200 num_viz(GL_PRIMITIVES_GENERATED) 200 viz_bytes 12800



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
#include <glm/glm.hpp>


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
        mat4 tr = ITransform[0] ;
        vec4 tla = tr[3] ; 

        if ( tla.x >= -2.f )   // arbitrary visibility criteria
        {    
            VizTransform0 = tr[0]  ;
            VizTransform1 = tr[1]  ;
            VizTransform2 = tr[2]  ;
            VizTransform3 = tr[3]  ;

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


struct Demo 
{
    static const unsigned QSIZE ; 

    Frame*  frame ;   
    Prog*   cull ; 
    Prog*   norm ; 
   
    unsigned num_viz ; 
    unsigned num_inst ; 
    unsigned num_vert ; 

    Transforms* tr ; 
    Transforms* trviz ;
 
    Buf* vbuf ; 
    Buf* ibuf ; 
    Buf* cbuf ; 
    Renderer* rdr ; 

    Demo(Prog* cull_, Prog* norm_);

    void init();
    void initVert();
    void initCull();
    void initNorm();
    void initBuffers();

    void instcull();
    void render();
    void renderLoop();

    void destroy();

};



Demo::Demo(Prog* cull_, Prog* norm_) 
    :
    frame(new Frame),
    cull(cull_),
    norm(norm_),
    num_viz(0),
    num_inst(10),
    num_vert(3),
    tr(new Transforms(num_inst, 4, 4, NULL)),
    trviz(new Transforms(num_inst, 4, 4, new float[num_inst*4*4])),
    vbuf(new Buf(sizeof(float)*4*num_vert, NULL )),  
    ibuf(new Buf(tr->num_bytes(), tr->itra )), 
    cbuf(new Buf(tr->num_bytes(), NULL )),
    rdr(new Renderer)
{
    init();
}

const unsigned Demo::QSIZE = 4*sizeof(float) ;

void Demo::destroy()
{
    rdr->destroy();
    cull->destroy();
    norm->destroy();
    frame->destroy();
}

void Demo::init()
{
    initVert();
    initCull();
    initNorm();
    initBuffers();
}

void Demo::initVert()
{
    struct V { float x,y,z,w ; };

    vbuf->ptr = new V[num_vert] ; 

    V* vptr = (V*)vbuf->ptr ; 

    vptr[0] = {  0.05f ,   0.05f,  0.00f,  1.f } ;
    vptr[1] = {  0.05f ,  -0.05f,  0.00f,  1.f } ;
    vptr[2] = {  0.00f ,   0.00f,  0.00f,  1.f } ;
}

void Demo::initCull()
{
    cull->compile();
    cull->create();

    // spacing the output from cull phase to write last vec4 of mat4
    //const char *vars[] = { "gl_SkipComponents4", "gl_SkipComponents4", "gl_SkipComponents4",  "CulledPosition" };
    const char *vars[] = { "VizTransform0", "VizTransform1", "VizTransform2",  "VizTransform3" };
    glTransformFeedbackVaryings(cull->program, 4, vars, GL_INTERLEAVED_ATTRIBS); 

    cull->link();
}

void Demo::initNorm()
{
    norm->compile();
    norm->create();
    norm->link();
}

void Demo::initBuffers()
{
    rdr->upload( ibuf , GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    rdr->upload( vbuf , GL_ARRAY_BUFFER, GL_STATIC_DRAW);


    //rdr->upload( cbuf , GL_ARRAY_BUFFER, GL_STREAM_DRAW);
    rdr->upload( cbuf , GL_ARRAY_BUFFER, GL_DYNAMIC_COPY);


}

void Demo::instcull()
{
    glUseProgram(cull->program);
    glBindVertexArray(rdr->vao);

    glBindBuffer(GL_ARRAY_BUFFER, ibuf->id);

    Att ci0(LOC_InstanceTransform + 0 , 4, 4*QSIZE, 0 ); 
    Att ci1(LOC_InstanceTransform + 1 , 4, 4*QSIZE, 1*QSIZE ); 
    Att ci2(LOC_InstanceTransform + 2 , 4, 4*QSIZE, 2*QSIZE ); 
    Att ci3(LOC_InstanceTransform + 3 , 4, 4*QSIZE, 3*QSIZE ); 

    glEnableVertexAttribArray(ci0.loc);
    glEnableVertexAttribArray(ci1.loc);
    glEnableVertexAttribArray(ci2.loc);
    glEnableVertexAttribArray(ci3.loc);

    GLuint query;
    glGenQueries(1, &query);


    unsigned num_tr = tr->num_items() ;

    glEnable(GL_RASTERIZER_DISCARD);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, cbuf->id );
    glBeginTransformFeedback(GL_POINTS);
    {
        glBeginQuery(GL_PRIMITIVES_GENERATED, query );
        {
            glDrawArrays(GL_POINTS, 0, num_tr );
        }
        glEndQuery(GL_PRIMITIVES_GENERATED); 
    }
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);

    glFlush();

    glGetQueryObjectuiv(query, GL_QUERY_RESULT, &num_viz);
   
    unsigned viz_bytes = num_viz*4*QSIZE ; 
    std::cout 
            << " num_tr " << num_tr 
            << " num_viz(GL_PRIMITIVES_GENERATED) " << num_viz 
            << " viz_bytes " << viz_bytes
            << std::endl 
            ; 

    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, viz_bytes, trviz->itra );

    if(num_viz > 0)
    {
        trviz->dump(num_viz);    
    }

    glUseProgram(0);

}



void Demo::render()
{
    glUseProgram(norm->program);
    glBindVertexArray(rdr->vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbuf->id);
    Att vp(LOC_VertexPosition, 4, QSIZE, 0); 
    glEnableVertexAttribArray(vp.loc);

    //bool use_cull = true ; 
    bool use_cull = false ; 
    unsigned num_draw = use_cull ? num_viz : num_inst ;

     
    std::cout << "Demo::render"
              << " num_inst " << num_inst
              << " num_viz " << num_viz
              << " num_draw " << num_draw
              << " use_cull " << use_cull 
              << std::endl
              ; 


    glBindBuffer(GL_ARRAY_BUFFER, use_cull ? cbuf->id : ibuf->id );    

    Att vip0(LOC_VizInstanceTransform + 0, 4, 4*QSIZE, 0*QSIZE ); 
    Att vip1(LOC_VizInstanceTransform + 1, 4, 4*QSIZE, 1*QSIZE ); 
    Att vip2(LOC_VizInstanceTransform + 2, 4, 4*QSIZE, 2*QSIZE ); 
    Att vip3(LOC_VizInstanceTransform + 3, 4, 4*QSIZE, 3*QSIZE ); 

    GLuint divisor = 1 ;   // number of instances between updates of ip.loc attribute , >1 will land that many instances on top of each other
    glVertexAttribDivisor(vip0.loc, divisor );
    glVertexAttribDivisor(vip1.loc, divisor );
    glVertexAttribDivisor(vip2.loc, divisor );
    glVertexAttribDivisor(vip3.loc, divisor );
    
    glEnableVertexAttribArray(vip0.loc);  
    glEnableVertexAttribArray(vip1.loc);  
    glEnableVertexAttribArray(vip2.loc);  
    glEnableVertexAttribArray(vip3.loc);  

    glDrawArraysInstanced( GL_TRIANGLES, 0, num_vert,  num_draw );
}

void Demo::renderLoop()
{
    unsigned count(0); 
    while (!glfwWindowShouldClose(frame->window) && count++ < 300 )
    {
        std::cout << "Demo::renderLoop count " << count << std::endl ; 

        int width, height;
        glfwGetFramebufferSize(frame->window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        instcull();
        render();

        glfwSwapBuffers(frame->window);
        glfwPollEvents();
    }
}


int main()
{
    Prog* cull = new Prog(vertCullSrc, geomCullSrc, NULL ) ; 
    Prog* norm = new Prog(vertNormSrc, NULL, fragNormSrc ) ; 
    Demo app(cull, norm) ; 
    app.renderLoop();      
    exit(EXIT_SUCCESS);
}


