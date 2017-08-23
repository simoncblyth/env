#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>


#include "Demo.hh"
#include "Prog.hh"
#include "Frame.hh"
#include "Tra.hh"
#include "Buf.hh"
#include "Geom.hh"
#include "Comp.hh"
#include "Cam.hh"
#include "Vue.hh"
#include "BB.hh"
#include "G.hh"




Demo::Demo() 
    :
    uniform(new Uniform),
    geom(new Geom(3,300)),
    comp(new Comp),
    frame(new Frame),
    cull(new Prog(vertCullSrc, geomCullSrc, NULL )), 
    draw(new Prog(vertDrawSrc, NULL, fragDrawSrc ))
{
    init();
}



const unsigned Demo::LOC_InstanceTransform = 0 ;  

const char* Demo::vertCullSrc = R"glsl(

    #version 400

    layout( location = 0) in mat4 InstanceTransform ;
    out mat4 ITransform ;       

    void main() 
    {      
        ITransform = InstanceTransform ; 
    }

)glsl";

const char* Demo::geomCullSrc = R"glsl(

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

        if ( tla.x > 0.f )   // arbitrary visibility criteria
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


const unsigned Demo::LOC_VertexPosition = 0 ;  
const unsigned Demo::LOC_VizInstanceTransform = 1 ;  

const char* Demo::vertDrawSrc = R"glsl(

    #version 400 core

    uniform MatrixBlock  
    {
        mat4 ModelView;
        mat4 ModelViewProjection;
    } ;

    layout (location = 0) in vec4 VertexPosition;
    layout (location = 1) in mat4 VizInstanceTransform ;
    void main()
    {
        gl_Position = ModelViewProjection * VizInstanceTransform * VertexPosition ;
        //gl_Position = VertexPosition ;
    }

)glsl";

const char* Demo::fragDrawSrc = R"glsl(

    #version 400 core 
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }

)glsl";





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



const unsigned Demo::QSIZE = 4*sizeof(float) ;

void Demo::destroy()
{
    cull->destroy();
    draw->destroy();
    frame->destroy();
}


void Demo::upload(Buf* buf)
{
    glGenBuffers(1, &buf->id);
    glBindBuffer(GL_ARRAY_BUFFER, buf->id);
    glBufferData(GL_ARRAY_BUFFER, buf->num_bytes, buf->ptr,  GL_STATIC_DRAW );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Demo::loadMeshData()
{
    Buf* vbuf = geom->vbuf ; 
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


    GLuint uniformBlockIndex = glGetUniformBlockIndex(draw->program, "MatrixBlock") ;
    assert(uniformBlockIndex != GL_INVALID_INDEX);

    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(draw->program, uniformBlockIndex,  uniformBlockBinding );


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
    // original instance transforms
    Buf* ibuf = geom->ibuf ; 

    glGenBuffers(1, &this->transformBO);
    glBindBuffer(GL_ARRAY_BUFFER, this->transformBO);
    glBufferData(GL_ARRAY_BUFFER, ibuf->num_bytes, ibuf->ptr, GL_STATIC_DRAW); 

    // allocate space for culled instance transforms

    glGenBuffers(1, &this->culledTransformBO);
    glBindBuffer(GL_ARRAY_BUFFER, this->culledTransformBO);
    glBufferData(GL_ARRAY_BUFFER, ibuf->num_bytes, NULL, GL_DYNAMIC_COPY); 

    // query for counting the surviving transforms

    glGenQueries(1, &this->culledTransformQuery);

    // vertex array for culling instance transforms, via vertex+geometry shader

    glGenVertexArrays(1, &cullVertexArray );
    glBindVertexArray(this->cullVertexArray );

    glBindBuffer(GL_ARRAY_BUFFER, this->transformBO); // original transforms fed in 

    glEnableVertexAttribArray(LOC_InstanceTransform + 0);
    glVertexAttribPointer(    LOC_InstanceTransform + 0, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(0*QSIZE) );

    glEnableVertexAttribArray(LOC_InstanceTransform + 1);
    glVertexAttribPointer(    LOC_InstanceTransform + 1, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(1*QSIZE) );

    glEnableVertexAttribArray(LOC_InstanceTransform + 2);
    glVertexAttribPointer(    LOC_InstanceTransform + 2, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(2*QSIZE) );

    glEnableVertexAttribArray(LOC_InstanceTransform + 3);
    glVertexAttribPointer(    LOC_InstanceTransform + 3, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(3*QSIZE) );

    errcheck("Demo::createInstances");
}



void Demo::setupUniformBuffer()
{
     // same UBO can be used from all shaders
    glGenBuffers(1, &this->uniformBO);
    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(Uniform), this->uniform, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint binding_point_index = 0 ;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point_index, this->uniformBO);
}


void Demo::updateUniform(float t)
{
//

    comp->update();
    uniform->ModelView = comp->world2eye ; 
    //uniform->ModelViewProjection = glm::transpose(comp->world2clip) ;  
    //uniform->ModelViewProjection = comp->world2clip ;  
    uniform->ModelViewProjection = comp->world2eye ;  

/*

    glm::vec3 tla(-0.25f,-0.25f,0.f);
    tla *= t ; 
    glm::mat4 m = glm::translate(glm::mat4(1.f), tla);
    uniform->ModelView = m ;
    uniform->ModelViewProjection = m  ;
*/

/*
    glm::mat4 m ; 

    m[0] = {  0.596, 0.000, 0.608, 0.607 } ;
    m[1] = { -0.456, 0.000, 0.795, 0.794 } ;
    m[2] = { 0.000, 1.000, 0.000, 0.000 } ;

    m[3] = {-0.5,0,0,2.5} ;

   0.794   0.000  -0.607   0.000 
                          -0.607   0.000  -0.794   0.000 
                           0.000   1.000   0.000   0.000 
                          -0.103   0.000  -1.778   1.000 




    m[0] = {   0.794,   0.000,  -0.607,   0.000,  } ; 
    m[1] = {  -0.607,   0.000,  -0.794,   0.000,  } ; 
    m[2] = {   0.000,   1.000,   0.000,   0.000,  } ; 
    m[3] = {  -0.103,   0.000,  -1.778,   3.000,  } ; 


    uniform->ModelViewProjection = m  ;



*/


    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Uniform), this->uniform);
}


void Demo::targetGeometry()
{
    BB* ibb = geom->ibb ; 
    glm::vec4 ice = ibb->get_center_extent();

    std::cout << "Demo::targetGeometry" 
              << G::gpresent("ice", ice)
              << std::endl
              ;

    comp->setCenterExtent(ice);      
    comp->vue->setEye(-1.3, -1.7, 0);      
    comp->cam->setFocus(ice.w, 100.f);      
    comp->update();
    comp->dump();

}




void Demo::init()
{
    setupUniformBuffer();

    loadShaders();

    loadMeshData();
    createInstances();
    targetGeometry();

    this->allVertexArray = createVertexArray(this->transformBO); 
    this->drawVertexArray = createVertexArray(this->culledTransformBO); 

    errcheck("Demo::init");
}

GLuint Demo::createVertexArray(GLuint instanceBO) 
{
    // instanced vertex array for rendering, incorporating 
    // both the mesh vertices and the transforms 

    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, this->vertexBO);
    glEnableVertexAttribArray(LOC_VertexPosition); 
    glVertexAttribPointer(LOC_VertexPosition, 4, GL_FLOAT, GL_FALSE, QSIZE, (void*)0);

    GLuint divisor = 1 ;   // number of instances between updates of attribute , >1 will land that many instances on top of each other
    glBindBuffer(GL_ARRAY_BUFFER, instanceBO);

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


void Demo::renderScene(float t)
{
    updateUniform(t);

    /////////// 1st pass : filter instance transforms into culledTransformBO via transform feedback 

    glUseProgram(cull->program);
    glBindVertexArray(this->cullVertexArray);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, this->culledTransformBO );

    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    {
        glBeginQuery(GL_PRIMITIVES_GENERATED, this->culledTransformQuery  );
        {
            glDrawArrays(GL_POINTS, 0, this->geom->num_inst );
        }
        glEndQuery(GL_PRIMITIVES_GENERATED); 
    }
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);

    glFlush();

    glGetQueryObjectuiv(this->culledTransformQuery, GL_QUERY_RESULT, &geom->num_viz);
  
    bool use_cull = true ; 

    std::cout 
          << " num_inst " << this->geom->num_inst 
          << " num_viz(GL_PRIMITIVES_GENERATED) " << this->geom->num_viz 
          << std::endl 
          ;
 
    /////////// 2nd pass : render just the surviving instances

    glUseProgram(draw->program);
    glBindVertexArray( use_cull ? this->drawVertexArray : this->allVertexArray);

    unsigned num_draw = use_cull ? geom->num_viz : geom->num_inst ; 

    if(num_draw > 0)    
    {  
        glDrawArraysInstanced( GL_TRIANGLES, 0, geom->num_vert,  num_draw );
        //pullback();
     }   
}

void Demo::pullback()
{
    GLenum target = GL_TRANSFORM_FEEDBACK_BUFFER ;

    if(geom->num_viz == 0) return ; 
    unsigned viz_bytes = geom->num_viz*4*QSIZE ; 
    glGetBufferSubData( target , 0, viz_bytes, geom->ctra );
    geom->ctra->dump(geom->num_viz);    
}


void Demo::renderLoop()
{
    unsigned count(0); 

    glEnable(GL_DEPTH_TEST);
    

    while (!glfwWindowShouldClose(frame->window) && count++ < 2000 )
    {
        frame->listen();

        //std::cout << "Demo::renderLoop count " << count << std::endl ; 

        glfwGetFramebufferSize(frame->window, &comp->cam->width, &comp->cam->height);
        glViewport(0, 0, comp->cam->width, comp->cam->height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        double t = glfwGetTime();

        renderScene((float)t);

        glfwSwapBuffers(frame->window);
    }
}


