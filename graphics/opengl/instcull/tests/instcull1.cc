// see  instcull-vi for notes

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


const unsigned LOC_VertexPosition = 0 ;  
const unsigned LOC_VizInstanceTransform = 1 ;  

const char* vertDrawSrc = R"glsl(

    #version 400 core
    layout (location = 0) in vec4 VertexPosition;
    layout (location = 1) in mat4 VizInstanceTransform ;
    void main()
    {
        gl_Position = VizInstanceTransform * VertexPosition ;
        //gl_Position = VertexPosition ;
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




// https://learnopengl.com/#!Getting-started/Camera

struct Camera
{
    glm::vec3 position ; 
    glm::vec3 front ; 
    glm::vec3 up ; 
    glm::mat4 view ; 
  
    Camera();
    void update();
};

Camera::Camera()
   :
   position(0,0,10),
   front(0,0,-1),
   up(0,1,0),
   view(1.)
{
   update();
}


void Camera::update()
{
   view = glm::lookAt( position , position + front , up );
}





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
    GLuint allVertexArray ;   

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
    void pullback(GLenum target);
    void destroy();

};

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

Demo::Demo(Prog* cull_, Prog* draw_) 
    :
    frame(new Frame),
    cull(cull_),
    draw(draw_),
    num_viz(0),
    num_inst(300),
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

/*
void Demo::upload(Buf* buf)
{
    glGenBuffers(1, &buf->id);
    glBindBuffer(GL_ARRAY_BUFFER, buf->id);
    glBufferData(GL_ARRAY_BUFFER, buf->num_bytes, buf->ptr,  GL_STATIC_DRAW );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
*/

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
    // original instance transforms

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

void Demo::init()
{
    loadShaders();

    loadMeshData();
    createInstances();

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


void Demo::renderScene()
{
    /////////// 1st pass : filter instance transforms into culledTransformBO via transform feedback 

    glUseProgram(cull->program);
    glBindVertexArray(this->cullVertexArray);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, this->culledTransformBO );

    glEnable(GL_RASTERIZER_DISCARD);
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
  
    bool use_cull = true ; 
/*
    std::cout 
          << " num_inst " << this->num_inst 
          << " num_viz(GL_PRIMITIVES_GENERATED) " << this->num_viz 
          << std::endl 
          ;
*/
 
    /////////// 2nd pass : render just the surviving instances

    glUseProgram(draw->program);
    glBindVertexArray( use_cull ? this->drawVertexArray : this->allVertexArray);

    unsigned num_draw = use_cull ? num_viz : num_inst ; 

    if(num_draw > 0)    
    {  
        glDrawArraysInstanced( GL_TRIANGLES, 0, num_vert,  num_draw );
        //pullback(GL_TRANSFORM_FEEDBACK_BUFFER);
     }   
}

void Demo::pullback(GLenum target)
{
    if(num_viz == 0) return ; 
    unsigned viz_bytes = num_viz*4*QSIZE ; 
    glGetBufferSubData( target , 0, viz_bytes, trviz->itra );
    trviz->dump(num_viz);    
}


void Demo::renderLoop()
{
    unsigned count(0); 
    while (!glfwWindowShouldClose(frame->window) && count++ < 2000 )
    {
        frame->listen();

        //std::cout << "Demo::renderLoop count " << count << std::endl ; 

        int width, height;
        glfwGetFramebufferSize(frame->window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        renderScene();

        glfwSwapBuffers(frame->window);
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


