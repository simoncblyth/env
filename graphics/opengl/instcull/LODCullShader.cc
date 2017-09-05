#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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
        
        //float f = 0.8f ; 
        float f = 1.0f ; 
        objectVisible = 
             ( IClip.x < IClip.w*f && IClip.x > -IClip.w*f  ) &&
             ( IClip.y < IClip.w*f && IClip.y > -IClip.w*f  ) &&
             ( IClip.z < IClip.w*f && IClip.z > -IClip.w*f  ) ? 1 : 0 ; 
        

        ITransform = InstanceTransform ; 
    }

)glsl";

const char* LODCullShader::geomSrc = R"glsl(

    #version 400
    // LODCullShader::geomSrc

    $UniformBlock 

    uniform vec4 CUT ; 

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

    layout(stream=2) out vec4 VizTransform0LOD2 ;
    layout(stream=2) out vec4 VizTransform1LOD2 ;
    layout(stream=2) out vec4 VizTransform2LOD2 ;
    layout(stream=2) out vec4 VizTransform3LOD2 ;


    void main() 
    {
        mat4 tr = ITransform[0] ;

        if(objectVisible[0] == 1)
        {    
            vec4 InstancePosition = tr[3] ; 
            vec4 IEye = ModelView * InstancePosition ;    
            float distance = length(IEye.xyz) ;

            int lod = distance < CUT.x ? 2 : ( distance < CUT.y ? 1 : 0 ) ;  

            switch(lod)
            {
               case 0:                       
                    VizTransform0LOD0 = tr[0]  ;
                    VizTransform1LOD0 = tr[1]  ;
                    VizTransform2LOD0 = tr[2]  ;
                    VizTransform3LOD0 = tr[3]  ;

                    EmitStreamVertex(0);
                    EndStreamPrimitive(0);
                    break ; 

               case 1:  
                    VizTransform0LOD1 = tr[0]  ;
                    VizTransform1LOD1 = tr[1]  ;
                    VizTransform2LOD1 = tr[2]  ;
                    VizTransform3LOD1 = tr[3]  ;

                    EmitStreamVertex(1);
                    EndStreamPrimitive(1);
                    break ; 

               case 2:  
                    VizTransform0LOD2 = tr[0]  ;
                    VizTransform1LOD2 = tr[1]  ;
                    VizTransform2LOD2 = tr[2]  ;
                    VizTransform3LOD2 = tr[3]  ;

                    EmitStreamVertex(2);
                    EndStreamPrimitive(2);
                    break ; 
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
    LOC_WORKAROUND(0),
    WORKAROUND(-1)
{
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
                           "VizTransform3LOD1",
                           "gl_NextBuffer",
                           "VizTransform0LOD2", 
                           "VizTransform1LOD2", 
                           "VizTransform2LOD2",  
                           "VizTransform3LOD2"
                         };
    glTransformFeedbackVaryings(prog->program, 14, vars, GL_INTERLEAVED_ATTRIBS); 

    glBindAttribLocation(prog->program, LOC_InstanceTransform, "InstanceTransform");
    prog->link();


    GLuint LOC_LodDistance = glGetUniformLocation(prog->program, "CUT" );
    glm::vec4 LodDistance(20.f, 100.f, 0.f, 0.f );
    // hmm if lowest LOD dist is less than near, never get to see the best level 
   
    glUniform4fv( LOC_LodDistance, 1, glm::value_ptr(LodDistance));



    GU::errchk("LODCullShader::init");
}


void LODCullShader::setupFork(Buf* src_, Buf4* dst_)
{
    // invoked from ICDemo::init, for each LOD level generate output count queries 
    // and bind tranform feedback stream output buffers, 
    // create single forking VAO

    src = src_ ; 
    dst = dst_ ;

    num_lod = dst->num_buf() ; 
    assert(num_lod <= LOD_MAX);

    std::cout << "LODCullShader::setupFork" 
              << " src->id " << src->id
              << " src->num_items " << src->num_items
              << " num_lod " << num_lod
              << std::endl ; 

    dst->dump();

    for(int i=0 ; i < num_lod ; i++) glGenQueries(1, &this->lodQuery[i]);

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





void LODCullShader::pullback()
{
    for(int i=0 ; i < num_lod ; i++)
    {
        Buf* tbuf = dst->at(i) ;

        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, tbuf->id ); // <-- without thus pullback same content each time
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbuf->num_bytes, tbuf->ptr );

        std::cout << "LODCullShader::pullback" << i << std::endl ; 
        tbuf->dump("LODCullShader::pullback");
        
    }
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

    for (int i=0; i< num_lod; i++) 
    {
        Buf* tbuf = dst->at(i) ;
        glGetQueryObjectiv(lodQuery[i], GL_QUERY_RESULT, &tbuf->query_count);
    }

   //  http://apprize.info/programming/opengl_1/13.html
   //
   //    querying will likely stall the pipeline
   //    to avoid that could check if the result is available 
   //    first with GL_QUERY_RESULT_AVAILABLE
   //    before making the actual query ...
   //
   //    The outcome of applyFork is updated instance buffers for 
   //    each LOD and corresponding counts..   Need to know
   //    the counts to properly use these buffers.
   //
   //    
   //    But how to organize deferred querying ?
   //
   //    Hmm would be complicated... would need to have a 2nd set of
   //    instance buffers and ping-pong between them ?

        
}


void LODCullShader::applyForkStreamQueryWorkaround()
{
/* 
    As investigated in tests/txfStream.cc

    Forking into 4 separate streams works, but the 
    query counts only work for stream 0 ? The others 
    yielding zero.

    Looks like a driver bug...

    This workaround run the transform 
    feedback again to get the counts for streams > 0

    In an attempt to minimize the time to do that 
    "devnull" zero-sized buffers are attached 
    to avoid data movement.

*/

    glUseProgram(this->prog->program);
    glBindVertexArray(this->forkVertexArray);

    int i0 = 1 ; 

    for (int i=i0; i< dst->num_buf() ; i++) 
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, dst->devnull->id );

    glEnable(GL_RASTERIZER_DISCARD);
    for(int i=i0 ; i < num_lod ; i++)
    {
        glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, i, this->lodQuery[i]  );
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, src->num_items );
        glEndTransformFeedback();
        glEndQueryIndexed(GL_PRIMITIVES_GENERATED, i );
    }
    glDisable(GL_RASTERIZER_DISCARD);

    for (int i=i0; i< num_lod; i++) 
    {
        Buf* tbuf = dst->at(i) ;
        glGetQueryObjectiv(lodQuery[i], GL_QUERY_RESULT, &tbuf->query_count);
    }

    for (int i=i0; i< dst->num_buf() ; i++) 
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, dst->at(i)->id );
         
}

void LODCullShader::dump(const char* msg)
{
    std::cout << msg 
              << " num_lod " << num_lod
              << " src " << src->brief()
              << " dst " << dst->desc() 
              << std::endl ; 
               ;
}






