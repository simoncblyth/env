#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GU.hh"
#include "Prog.hh"
#include "Buf.hh"
#include "Buf4.hh"
#include "SContext.hh"
#include "LODCullShader.hh"


const unsigned LODCullShader::LOC_InstanceTransform = 0 ;  

const char* LODCullShader::vertSrc = R"glsl(

    #version 400
    // LODCullShader::vertSrc

    $UniformBlock 

    layout( location = 0) in mat4 InstanceTransform ;
    out mat4 ITransform ;       
    flat out int objectVisible;

    void main() 
    {      
        vec4 InstancePosition = InstanceTransform[3] ; 
        vec4 IClip = ModelViewProjection * InstancePosition ;    
        
        float f = 0.8f ; 
        //float f = 1.0f ; 
        objectVisible = 
             ( IClip.x < IClip.w*f && IClip.x > -IClip.w*f  ) &&
             ( IClip.y < IClip.w*f && IClip.y > -IClip.w*f  ) &&
             ( IClip.z < IClip.w*f && IClip.z > -IClip.w*f  ) ? 1 : 0 ; 
        
        //objectVisible = InstancePosition.z < 0.f ? 1 : 0 ;  // arbitrary visibility criteria

        ITransform = InstanceTransform ; 
    }

)glsl";

const char* LODCullShader::geomSrc = R"glsl(

    #version 400
    // LODCullShader::geomSrc

    $UniformBlock 


    layout(points) in; 
    layout(points, max_vertices = 1) out;
    
    in mat4 ITransform[1] ;
    flat in int objectVisible[1]; 

    layout(stream=0) out vec4 VizTransform0LOD0 ;
    layout(stream=0) out vec4 VizTransform1LOD0 ;
    layout(stream=0) out vec4 VizTransform2LOD0 ;
    layout(stream=0) out vec4 VizTransform3LOD0 ;

    layout(stream=1) out vec4 VizTransform0LOD1 ;
    layout(stream=1) out vec4 VizTransform1LOD1 ;
    layout(stream=1) out vec4 VizTransform2LOD1 ;
    layout(stream=1) out vec4 VizTransform3LOD1 ;


    void main() 
    {
        mat4 tr = ITransform[0] ;

        if(objectVisible[0] == 1)
        {    
            vec4 InstancePosition = tr[3] ; 
            vec4 IEye = ModelView * InstancePosition ;    
            float distance = IEye.y ;

            if( distance < 0.f ) 
            {
                VizTransform0LOD0 = tr[0]  ;
                VizTransform1LOD0 = tr[1]  ;
                VizTransform2LOD0 = tr[2]  ;
                VizTransform3LOD0 = tr[3]  ;

                EmitStreamVertex(0);
                EndStreamPrimitive(0);
            }   
            else
            {
                VizTransform0LOD1 = tr[0]  ;
                VizTransform1LOD1 = tr[1]  ;
                VizTransform2LOD1 = tr[2]  ;
                VizTransform3LOD1 = tr[3]  ;

                EmitStreamVertex(1);
                EndStreamPrimitive(1);
            }
        }
    }

)glsl";

const unsigned LODCullShader::QSIZE = sizeof(float)*4 ; 

LODCullShader::LODCullShader(SContext* context_)
    :
    context(context_),
    prog(new Prog(
                SContext::ReplaceUniformBlockToken(vertSrc), 
                SContext::ReplaceUniformBlockToken(geomSrc), 
                NULL )),
    src(NULL),
    dst(NULL),
    num_lod(0),
    num_viz(0)
{
    for(unsigned i=0 ; i < LOD_MAX ; i++ ) lodCount[i] = 0u ; 

    init();
}

void LODCullShader::destroy()
{
    prog->destroy();
}

void LODCullShader::init()
{
    prog->compile();
    prog->create();
    const char *vars[] = { 
                           "VizTransform0LOD0", 
                           "VizTransform1LOD0", 
                           "VizTransform2LOD0",  
                           "VizTransform3LOD0",
                           "gl_NextBuffer",
                           "VizTransform0LOD1", 
                           "VizTransform1LOD1", 
                           "VizTransform2LOD1",  
                           "VizTransform3LOD1"
                         };
    glTransformFeedbackVaryings(prog->program, 9, vars, GL_INTERLEAVED_ATTRIBS); 

    glBindAttribLocation(prog->program, LOC_InstanceTransform, "InstanceTransform");
    prog->link();

    GU::errchk("LODCullShader::init");
}


void LODCullShader::setupFork(Buf* src_, Buf4* dst_)
{
    src = src_ ; 
    dst = dst_ ;

    num_lod = dst->num_buf() ; 
    assert(num_lod <= LOD_MAX);

    std::cout << "LODCullShader::setupFork" 
              << " src->id " << src->id
              << " src->num_items " << src->num_items
              << " num_lod " << num_lod
              << std::endl ; 


    for(int i=0 ; i < num_lod ; i++) 
    {
         std::cout << " i " << i 
                   << " dst->at(i)->id  " << dst->at(i)->id 
                   << std::endl ; 
 
    }

    for(int i=0 ; i < num_lod ; i++) glGenQueries(1, &this->lodQuery[i]);

    //glGenQueries(num_lod, this->lodQuery);

    forkVertexArray = createForkVertexArray(src->id);

    for (int i=0; i< num_lod; i++) 
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, dst->at(i)->id );

}


GLuint LODCullShader::createForkVertexArray(GLuint instanceBO) 
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

    GU::errchk("LODCullShader::createForkVertexArray");
    return vertexArray;
}



void LODCullShader::applyFork()
{
    // http://rastergrid.com/blog/2010/10/gpu-based-dynamic-geometry-lod/
    assert(src);

    glUseProgram(this->prog->program);
    glBindVertexArray(this->forkVertexArray);
    glEnable(GL_RASTERIZER_DISCARD);

    for(int i=0 ; i < num_lod ; i++)
        glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, i, this->lodQuery[i]  );

    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, src->num_items );
    glEndTransformFeedback();

    for(int i=0 ; i < num_lod ; i++)
        glEndQueryIndexed(GL_PRIMITIVES_GENERATED, i );

    glDisable(GL_RASTERIZER_DISCARD);

    glFlush();

    for (int i=0; i< num_lod; i++) 
        glGetQueryObjectiv(lodQuery[i], GL_QUERY_RESULT, &this->lodCount[i]);


    std::cout << "LODCullShader::applyFork"
              << " num_lod " << num_lod
              << " src->num_items " << src->num_items
              << " lodCount[0] " << lodCount[0]
              << " lodCount[1] " << lodCount[1]
              << " lodCount[2] " << lodCount[2]
              << " lodCount[3] " << lodCount[3]
              << std::endl ; 


}



