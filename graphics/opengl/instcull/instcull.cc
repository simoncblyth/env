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
//#include "Renderer.hh"
//#include "Att.hh"
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

        if ( tla.x > -2.f )   // arbitrary visibility criteria
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

const char* vertDrawSrc = R"glsl(

    #version 400 core
    layout (location = 0) in vec4 VertexPosition;
    layout (location = 1) in mat4 VizInstanceTransform ;
    void main()
    {
        //gl_Position = VizInstanceTransform * VertexPosition ;
        gl_Position = VertexPosition ;
    }

)glsl";

const char* fragDrawSrc = R"glsl(

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
    Prog*   draw ; 
   
    unsigned num_viz ; 
    unsigned num_inst ; 
    unsigned num_vert ; 

    Transforms* tr ; 
    Transforms* trviz ;
 
    Buf* vbuf ; 
    Buf* ibuf ; 
    Buf* cbuf ; 

    GLuint cullVertexArray ;   
    GLuint drawVertexArray ;   

    GLuint vertexBO ;   
    GLuint transformBO ;   

    GLuint culledTransformBO ;   
    GLuint culledTransformQuery ;   


    Demo(Prog* cull_, Prog* draw_);

    void init();
    void loadMeshData();
    void loadShaders();
    void createInstances();
    void renderScene();
    void renderLoop();
    void errcheck(const char* msg );
    GLuint createVertexArray(GLuint instanceBO) ;

    //void instcull();
    //void render();

    void destroy();

};



Demo::Demo(Prog* cull_, Prog* draw_) 
    :
    frame(new Frame),
    cull(cull_),
    draw(draw_),
    num_viz(0),
    num_inst(10),
    num_vert(3),
    tr(new Transforms(num_inst, 4, 4, NULL)),
    trviz(new Transforms(num_inst, 4, 4, new float[num_inst*4*4])),
    vbuf(new Buf(QSIZE*num_vert, NULL )),  
    ibuf(new Buf(tr->num_bytes(), tr->itra )), 
    cbuf(new Buf(tr->num_bytes(), NULL ))
{
    init();
}

const unsigned Demo::QSIZE = 4*sizeof(float) ;

void Demo::destroy()
{
    cull->destroy();
    draw->destroy();
    frame->destroy();
}

void Demo::init()
{
    loadShaders();

    loadMeshData();
    createInstances();

    this->drawVertexArray = createVertexArray(this->culledTransformBO); 

    errcheck("Demo::init");
}

void Demo::loadMeshData()
{
    struct V { float x,y,z,w ; };

    vbuf->ptr = new V[num_vert] ; 

    V* vptr = (V*)vbuf->ptr ; 

    vptr[0] = {  0.05f ,   0.05f,  0.00f,  1.f } ;
    vptr[1] = {  0.05f ,  -0.05f,  0.00f,  1.f } ;
    vptr[2] = {  0.00f ,   0.00f,  0.00f,  1.f } ;

    glGenBuffers(1, &this->vertexBO);
    glBindBuffer(GL_ARRAY_BUFFER, this->vertexBO);
    glBufferData(GL_ARRAY_BUFFER, vbuf->num_bytes, vbuf->ptr, GL_STATIC_DRAW);

    errcheck("Demo::loadMeshData");
}


void Demo::loadShaders()
{
    draw->compile();
    draw->create();
    glBindAttribLocation(draw->program, LOC_VertexPosition , "VertexPosition");
    glBindAttribLocation(draw->program, LOC_VizInstanceTransform , "VizInstanceTransform");
    draw->link();

    cull->compile();
    cull->create();
    //const char *vars[] = { "gl_SkipComponents4", "gl_SkipComponents4", "gl_SkipComponents4",  "CulledPosition" };
    const char *vars[] = { "VizTransform0", "VizTransform1", "VizTransform2",  "VizTransform3" };
    glTransformFeedbackVaryings(cull->program, 4, vars, GL_INTERLEAVED_ATTRIBS); 

    glBindAttribLocation(cull->program, LOC_InstanceTransform, "InstanceTransform");

    cull->link();

    errcheck("Demo::loadShaders");
}

void Demo::createInstances()
{
    glGenBuffers(1, &this->transformBO);
    glBindBuffer(GL_ARRAY_BUFFER, this->transformBO);
    glBufferData(GL_ARRAY_BUFFER, ibuf->num_bytes, ibuf->ptr, GL_STATIC_DRAW); 

    glGenVertexArrays(1, &cullVertexArray );
    glBindVertexArray(this->cullVertexArray );

    glBindBuffer(GL_ARRAY_BUFFER, this->transformBO);

    glEnableVertexAttribArray(LOC_InstanceTransform + 0);
    glVertexAttribPointer(    LOC_InstanceTransform + 0, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(0*QSIZE) );

    glEnableVertexAttribArray(LOC_InstanceTransform + 1);
    glVertexAttribPointer(    LOC_InstanceTransform + 1, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(1*QSIZE) );

    glEnableVertexAttribArray(LOC_InstanceTransform + 2);
    glVertexAttribPointer(    LOC_InstanceTransform + 2, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(2*QSIZE) );

    glEnableVertexAttribArray(LOC_InstanceTransform + 3);
    glVertexAttribPointer(    LOC_InstanceTransform + 3, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(3*QSIZE) );


    glGenBuffers(1, &this->culledTransformBO);
    glBindBuffer(GL_ARRAY_BUFFER, this->culledTransformBO);
    glBufferData(GL_ARRAY_BUFFER, ibuf->num_bytes, NULL, GL_DYNAMIC_COPY); 

    glGenQueries(1, &this->culledTransformQuery);

    errcheck("Demo::createInstances");
}


GLuint Demo::createVertexArray(GLuint instanceBO) 
{
    GLuint vertexArray;

    glGenVertexArrays(1, &vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, this->vertexBO);
    glEnableVertexAttribArray(LOC_VertexPosition); 
    glVertexAttribPointer(LOC_VertexPosition, 4, GL_FLOAT, GL_FALSE, QSIZE, (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, instanceBO);

    GLuint divisor = 1 ;   // number of instances between updates of attribute , >1 will land that many instances on top of each other

    glEnableVertexAttribArray(LOC_VizInstanceTransform + 0);
    glVertexAttribPointer(    LOC_VizInstanceTransform + 0, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(0*QSIZE) );
    glVertexAttribDivisor(    LOC_VizInstanceTransform + 0, divisor );

    glEnableVertexAttribArray(LOC_VizInstanceTransform + 1);
    glVertexAttribPointer(    LOC_VizInstanceTransform + 1, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(1*QSIZE) );
    glVertexAttribDivisor(    LOC_VizInstanceTransform + 1, divisor );

    glEnableVertexAttribArray(LOC_VizInstanceTransform + 2);
    glVertexAttribPointer(    LOC_VizInstanceTransform + 2, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(2*QSIZE) );
    glVertexAttribDivisor(    LOC_VizInstanceTransform + 2, divisor );

    glEnableVertexAttribArray(LOC_VizInstanceTransform + 3);
    glVertexAttribPointer(    LOC_VizInstanceTransform + 3, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(3*QSIZE) );
    glVertexAttribDivisor(    LOC_VizInstanceTransform + 3, divisor );

    errcheck("Demo::createVertexArray");

    return vertexArray;
}


void Demo::errcheck(const char* msg)
{
    GLenum glError;
    if ((glError = glGetError()) != GL_NO_ERROR) 
     {
        std::cout 
            << msg 
            << " : Warning: OpenGL error code: " 
            << glError 
            << std::endl
            ;
    }   
}




void Demo::renderScene()
{

    glUseProgram(cull->program);
    glEnable(GL_RASTERIZER_DISCARD);

    glBindVertexArray(this->cullVertexArray);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, this->culledTransformBO );
    glBeginTransformFeedback(GL_POINTS);
    {
        glBeginQuery(GL_PRIMITIVES_GENERATED, this->culledTransformQuery  );
        {
            glDrawArrays(GL_POINTS, 0, this->num_inst );
        }
        glEndQuery(GL_PRIMITIVES_GENERATED); 
    }
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);

    glFlush();

    glGetQueryObjectuiv(this->culledTransformQuery, GL_QUERY_RESULT, &num_viz);
  


    //num_viz = num_inst ;  

    glUseProgram(draw->program);

    glBindVertexArray(this->drawVertexArray);


        std::cout 
                << " num_inst " << this->num_inst 
                << " num_viz(GL_PRIMITIVES_GENERATED) " << this->num_viz 
                << std::endl 
                ;
 

    if(num_viz > 0)
    {
        unsigned viz_bytes = num_viz*4*QSIZE ; 
        std::cout 
                << " num_inst " << this->num_inst 
                << " num_viz(GL_PRIMITIVES_GENERATED) " << this->num_viz 
                << " viz_bytes " << viz_bytes
                << std::endl 
                ; 

        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, viz_bytes, trviz->itra );
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, viz_bytes, trviz->itra );

        trviz->dump(num_viz);    

        glDrawArraysInstanced( GL_TRIANGLES, 0, num_vert,  num_viz );
    }
}

void Demo::renderLoop()
{
    unsigned count(0); 
    while (!glfwWindowShouldClose(frame->window) && count++ < 10 )
    {
        std::cout << "Demo::renderLoop count " << count << std::endl ; 

        int width, height;
        glfwGetFramebufferSize(frame->window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        renderScene();

        glfwSwapBuffers(frame->window);
        glfwPollEvents();
    }
}

int main()
{
    Prog* cull = new Prog(vertCullSrc, geomCullSrc, NULL ) ; 
    Prog* draw = new Prog(vertDrawSrc, NULL, fragDrawSrc ) ; 
    Demo app(cull, draw) ; 
    app.renderLoop();      
    exit(EXIT_SUCCESS);
}


