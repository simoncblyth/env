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
#include "GU.hh"


/*

NEXT:

* generalize as mockup some more realistic geometry , eg a big sphere of many sphere instances of two types
* frustum culling 
* lod streaming, starting with 2 lod levels : original and bbox

*/

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

const unsigned Demo::QSIZE = 4*sizeof(float) ;

void Demo::init()
{
    setupUniformBuffer();
    loadShaders();

    loadMeshData(geom->vbuf);
    createInstances(geom->ibuf);
    targetGeometry(geom->ibb);

    this->allVertexArray = createInstancedRenderVertexArray(this->transformBO); 
    this->drawVertexArray = createInstancedRenderVertexArray(this->culledTransformBO); 

    GU::errchk("Demo::init");
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
    float angle = t ; 
    comp->vue->setEye( glm::cos(angle), 1, glm::sin(angle) );
    comp->update();

    uniform->ModelView = comp->world2eye ;  
    uniform->ModelViewProjection = comp->world2clip ;  

    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Uniform), this->uniform);
}

void Demo::loadShaders()
{
    draw->compile();
    draw->create();
    glBindAttribLocation(draw->program, LOC_VertexPosition , "VertexPosition");
    glBindAttribLocation(draw->program, LOC_VizInstanceTransform , "VizInstanceTransform");
    draw->link();
    GU::errchk("Demo::loadShaders.draw");

    GLuint uniformBlockIndex = glGetUniformBlockIndex(draw->program, "MatrixBlock") ;
    assert(uniformBlockIndex != GL_INVALID_INDEX && "NB must use the uniform otherwise it gets optimized away") ;
    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(draw->program, uniformBlockIndex,  uniformBlockBinding );


    cull->compile();
    cull->create();
    const char *vars[] = { "VizTransform0", "VizTransform1", "VizTransform2",  "VizTransform3" };
    glTransformFeedbackVaryings(cull->program, 4, vars, GL_INTERLEAVED_ATTRIBS); 
    glBindAttribLocation(cull->program, LOC_InstanceTransform, "InstanceTransform");
    cull->link();

    GU::errchk("Demo::loadShaders.cull");
}

void Demo::upload(Buf* buf)
{
    glGenBuffers(1, &buf->id);
    glBindBuffer(GL_ARRAY_BUFFER, buf->id);
    glBufferData(GL_ARRAY_BUFFER, buf->num_bytes, buf->ptr,  GL_STATIC_DRAW );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Demo::loadMeshData(Buf* vbuf)
{
    glGenBuffers(1, &this->vertexBO);
    glBindBuffer(GL_ARRAY_BUFFER, this->vertexBO);
    glBufferData(GL_ARRAY_BUFFER, vbuf->num_bytes, vbuf->ptr, GL_STATIC_DRAW);

    GU::errchk("Demo::loadMeshData");
}

void Demo::createInstances(Buf* ibuf)
{
    glGenBuffers(1, &this->transformBO);
    glBindBuffer(GL_ARRAY_BUFFER, this->transformBO);
    glBufferData(GL_ARRAY_BUFFER, ibuf->num_bytes, ibuf->ptr, GL_STATIC_DRAW); 

    // allocate space for culled instance transforms
    glGenBuffers(1, &this->culledTransformBO);
    glBindBuffer(GL_ARRAY_BUFFER, this->culledTransformBO);
    glBufferData(GL_ARRAY_BUFFER, ibuf->num_bytes, NULL, GL_DYNAMIC_COPY); 

    // query for counting the surviving transforms
    glGenQueries(1, &this->culledTransformQuery);

    cullVertexArray = createTransformCullVertexArray(this->transformBO, LOC_InstanceTransform );

    GU::errchk("Demo::createInstances");
}

GLuint Demo::createInstancedRenderVertexArray(GLuint instanceBO) 
{
    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, this->vertexBO);
    glEnableVertexAttribArray(LOC_VertexPosition); 
    glVertexAttribPointer(LOC_VertexPosition, 4, GL_FLOAT, GL_FALSE, QSIZE, (void*)0);

    GLuint divisor = 1 ;   // number of instances between updates of attribute , >1 will land that many instances on top of each other
    GLuint loc =  LOC_VizInstanceTransform ;

    glBindBuffer(GL_ARRAY_BUFFER, instanceBO);

    glEnableVertexAttribArray(loc + 0);
    glVertexAttribPointer(    loc + 0, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(0*QSIZE) );
    glVertexAttribDivisor(    loc + 0, divisor );

    glEnableVertexAttribArray(loc + 1);
    glVertexAttribPointer(    loc + 1, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(1*QSIZE) );
    glVertexAttribDivisor(    loc + 1, divisor );

    glEnableVertexAttribArray(loc + 2);
    glVertexAttribPointer(    loc + 2, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(2*QSIZE) );
    glVertexAttribDivisor(    loc + 2, divisor );

    glEnableVertexAttribArray(loc + 3);
    glVertexAttribPointer(    loc + 3, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(3*QSIZE) );
    glVertexAttribDivisor(    loc + 3, divisor );

    GU::errchk("Demo::createInstancedRenderVertexArray");

    return vertexArray;
}


GLuint Demo::createTransformCullVertexArray(GLuint instanceBO, GLuint loc) 
{
    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, instanceBO); // original transforms fed in 

    glEnableVertexAttribArray(loc + 0);
    glVertexAttribPointer(    loc + 0, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(0*QSIZE) );

    glEnableVertexAttribArray(loc + 1);
    glVertexAttribPointer(    loc + 1, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(1*QSIZE) );

    glEnableVertexAttribArray(loc + 2);
    glVertexAttribPointer(    loc + 2, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(2*QSIZE) );

    glEnableVertexAttribArray(loc + 3);
    glVertexAttribPointer(    loc + 3, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(3*QSIZE) );

    // NB no divisor, are accessing instance transforms in a non-instanced manner to do the culling 

    GU::errchk("Demo::createTransformCullVertexArray");
    return vertexArray;
}


void Demo::targetGeometry(BB* bb)
{
    glm::vec4 ce = bb->get_center_extent();

    std::cout << "Demo::targetGeometry" 
              << G::gpresent("ce", ce)
              << std::endl
              ;

    comp->setCenterExtent(ce);      
    comp->vue->setEye( -0.01, 1, 1);       // avoid zeros, tend to cause no-viz geometry 
    comp->cam->setFocus( ce.w, 10.f);      
    comp->update();
    comp->dump();
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
  
    bool use_cull = false ; 

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

void Demo::destroy()
{
    cull->destroy();
    draw->destroy();
    frame->destroy();
}


