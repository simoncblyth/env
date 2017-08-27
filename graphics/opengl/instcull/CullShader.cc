
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GU.hh"
#include "Prog.hh"
#include "Buf.hh"
#include "SContext.hh"
#include "CullShader.hh"


const unsigned CullShader::LOC_InstanceTransform = 0 ;  

const char* CullShader::vertSrc = R"glsl(

    #version 400
    // CullShader::vertSrc

    $UniformBlock 

    layout( location = 0) in mat4 InstanceTransform ;
    out mat4 ITransform ;       
    flat out int objectVisible;

    void main() 
    {      
        vec4 InstancePosition = InstanceTransform[3] ; 
        vec4 IClip = ModelViewProjection * InstancePosition ;    
        
        //float f = 0.8f ; 
        float f = 1.0f ; 
        objectVisible = 
             ( IClip.x < IClip.w*f && IClip.x > -IClip.w*f  ) &&
             ( IClip.y < IClip.w*f && IClip.y > -IClip.w*f  ) &&
             ( IClip.z < IClip.w*f && IClip.z > -IClip.w*f  ) ? 1 : 0 ; 
        
        //objectVisible = InstancePosition.z < 0.f ? 1 : 0 ;  // arbitrary visibility criteria

        ITransform = InstanceTransform ; 
    }

)glsl";

const char* CullShader::geomSrc = R"glsl(

    #version 400

    layout(points) in; 
    layout(points, max_vertices = 1) out;
    
    in mat4 ITransform[1] ;
    flat in int objectVisible[1]; 

    out vec4 VizTransform0 ;
    out vec4 VizTransform1 ;
    out vec4 VizTransform2 ;
    out vec4 VizTransform3 ;

    void main() 
    {
        mat4 tr = ITransform[0] ;

        if(objectVisible[0] == 1)
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

const unsigned CullShader::QSIZE = sizeof(float)*4 ; 

CullShader::CullShader(SContext* context_)
    :
    context(context_),
    prog(new Prog(SContext::ReplaceUniformBlockToken(vertSrc), geomSrc, NULL )),
    src(NULL),
    dst(NULL)
{
    init();
}

void CullShader::destroy()
{
    prog->destroy();
}

void CullShader::init()
{
    prog->compile();
    prog->create();
    const char *vars[] = { "VizTransform0", "VizTransform1", "VizTransform2",  "VizTransform3" };
    glTransformFeedbackVaryings(prog->program, 4, vars, GL_INTERLEAVED_ATTRIBS); 
    glBindAttribLocation(prog->program, LOC_InstanceTransform, "InstanceTransform");
    prog->link();

    GU::errchk("CullShader::init");
}


void CullShader::setupTransformFilter(Buf* src_)
{
    src = src_ ; 

    // query for counting the surviving transforms
    glGenQueries(1, &this->culledTransformQuery);
    cullVertexArray = createTransformCullVertexArray(src->id);
}


void CullShader::applyTransformFilter(Buf* dst)
{
    assert(src);

    glUseProgram(this->prog->program);
    glBindVertexArray(this->cullVertexArray);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, dst->id );

    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    {
        glBeginQuery(GL_PRIMITIVES_GENERATED, this->culledTransformQuery  );
        {
            glDrawArrays(GL_POINTS, 0, src->num_items );
        }
        glEndQuery(GL_PRIMITIVES_GENERATED); 
    }
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);

    //glFlush();

    glGetQueryObjectiv(this->culledTransformQuery, GL_QUERY_RESULT, &dst->query_count );
}




GLuint CullShader::createTransformCullVertexArray(GLuint instanceBO) 
{
    GLuint loc = LOC_InstanceTransform ;

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

    GU::errchk("CullShader::createTransformCullVertexArray");
    return vertexArray;
}


