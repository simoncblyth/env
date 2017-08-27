/*

        LodDistance       3.000   5.000   7.000   0.000 
 feedback 0 (          1          2          0          0          0          0          0          0          0          0  ) 
 feedback 1 (        103        104          0          0          0          0          0          0          0          0  ) 
 feedback 2 (        205        206          0          0          0          0          0          0          0          0  ) 
 feedback 3 (        307        308        309        310          0          0          0          0          0          0  ) 
 qiv 0
 qiv 32
 qiv 0
 qiv 32
 count[ 0] = 2 count[ 1] = 2    <<<< 
 count[ 2] = 0 count[ 3] = 0
 count[ 4] = 0 count[ 5] = 0
 count[ 6] = 0 count[ 7] = 0


As shown by above feedback pullbacks the LOD forking into 4 separate streams works, 
but the query counts only work for stream 0 ? The others yielding zero.


Looks like a driver bug...

Possible workaround is to run the transform 
feedback as many times as the stream count  
BUT with empty attached buffers using a
uniform to control permuting of the stream 
predicates down into slot 0 that does return 
a count.




Initially did this:
 
    Workaround failure of glGetQueryIndexed for non-zero stream index by arranging the 
    desired count via SLOT uniform to be in stream 0 (the only one that succeeds to count). 
    Unfortunately will need to repeat the transform feedback for each slot, however
    can do this with a null TBO buffer attached to avoid any data movement.

But the issue appears to be with having multiple active queries, not with 
the non-zero stream index, so repeating the transform feedback for each index works. 
               
Could just use interop buffer and CUDA ? To do all the counts 
... hmm actually very simple task, essentially just histogramming a value  
into 2~4 bins : could do with thrust. Check throgl- 
            
            







* http://www.g-truc.net/post-0373.html



[Mesa-dev] i965: Implement GL_PRIMITIVES_GENERATED with non-zero streams

* https://patchwork.freedesktop.org/patch/33317/

*/

#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

#include "Frame.hh"
#include "Prog.hh"
#include "G.hh"

const unsigned LOC_inValue = 0 ;  

const GLchar* vertSrc = R"glsl(

    #version 400 core

    layout ( location = 0 ) in float inValue;
    out float geoValue;
    void main()
    {
        geoValue = inValue ;
    }

)glsl";

const GLchar* geomSrc = R"glsl(

    #version 410 core

    layout(points) in;
    layout(points, max_vertices = 1) out;

    uniform vec4 CUT ; 
    uniform int SLOT ; 

    in float[] geoValue;

    layout (stream=0) out float outValue0 ;
    layout (stream=1) out float outValue1 ;
    layout (stream=2) out float outValue2 ;
    layout (stream=3) out float outValue3 ;

    void main()
    {
        int lod = geoValue[0] < CUT.x ? 0 : ( geoValue[0] < CUT.y ? 1 : ( geoValue[0] < CUT.z ? 2 : 3 )) ;      

        if( SLOT > -1 ) 
        {
            if( lod == SLOT )  
            { 
                outValue0 =  SLOT*100.f + geoValue[0] ;
                EmitStreamVertex(0); 
            }
        }         
        else    // standard stream fork when SLOT == -1  
        {
            if( lod == 0 )
            {
                outValue0 =  0.f + geoValue[0] ;
                EmitStreamVertex(0);
            } 
            else if( lod == 1 )
            {
                outValue1 = 100.f + geoValue[0] ;
                EmitStreamVertex(1);
            }
            else if( lod == 2 )
            {
                outValue2 = 200.f + geoValue[0] ;
                EmitStreamVertex(2);
            }
            else
            {
                outValue3 = 300.f + geoValue[0] ;
                EmitStreamVertex(3);
            }
        }

    }

)glsl";



int main(int, char** argv)
{
    Frame frame(argv[0]) ; 

    Prog* prog = new Prog(vertSrc, geomSrc, NULL );

    prog->compile();
    prog->create();


    const GLchar* feedbackVaryings[] = { 
                        "outValue0", 
                        "gl_NextBuffer", 
                        "outValue1", 
                        "gl_NextBuffer", 
                        "outValue2", 
                        "gl_NextBuffer", 
                        "outValue3" };

    glTransformFeedbackVaryings(prog->program, 7, feedbackVaryings, GL_INTERLEAVED_ATTRIBS );


/*
    const GLchar* feedbackVaryings[] = { 
                        "outValue0", 
                        "outValue1", 
                        "outValue2", 
                        "outValue3" };
 
    glTransformFeedbackVaryings(prog->program, 4, feedbackVaryings, GL_SEPARATE_ATTRIBS );
*/


    glBindAttribLocation(prog->program, LOC_inValue , "inValue");

    prog->link();


    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    static const unsigned N = 10 ; 
    GLfloat data[N] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.f, 8.f, 9.f, 10.f  };

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(LOC_inValue);
    glVertexAttribPointer(LOC_inValue, 1, GL_FLOAT, GL_FALSE, 0, 0);


    enum { NUM = 4 } ;
    GLuint tbo[NUM];
    glGenBuffers(NUM, tbo);

    GLuint tboDEVNULL ;
    glGenBuffers(1, &tboDEVNULL);



    // needs to be done before GL_TRANSFORM_FEEDBACK_BUFFER  binding 
    GLuint txf ; 
    glGenTransformFeedbacks(1, &txf);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, txf );

    GLuint query[2*NUM];
    GLint count[2*NUM];
    glGenQueries(2*NUM, query);

    GLuint query2[2*NUM];
    GLint count2[2*NUM];
    glGenQueries(2*NUM, query2);


    //GLenum ttarget = GL_ARRAY_BUFFER ;
    GLenum ttarget = GL_TRANSFORM_FEEDBACK_BUFFER ;  
    for(int i=0 ; i < NUM ; i++)
    {
        glBindBuffer(ttarget, tbo[i]);
        glBufferData(ttarget, sizeof(data), nullptr, GL_DYNAMIC_COPY);
    }        

    glBindBuffer(ttarget, tboDEVNULL);
    unsigned size = 1 ; 
    glBufferData(ttarget, size, nullptr, GL_DYNAMIC_COPY);
 


    for(int i=0 ; i < NUM ; i++) glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, tbo[i]);

    glUseProgram(prog->program);

    GLuint LOC_LodDistance = glGetUniformLocation(prog->program, "CUT" );
    glm::vec4 LodDistance(3.f, 5.f, 7.f, 0.f );

    std::cout << G::gpresent("LodDistance", LodDistance) << std::endl ; 

    glUniform4fv( LOC_LodDistance, 1, glm::value_ptr(LodDistance));

    GLuint LOC_SLOT = glGetUniformLocation(prog->program, "SLOT" );
    GLint SLOT = -1 ; 
    glUniform1i( LOC_SLOT, SLOT );

    glEnable(GL_RASTERIZER_DISCARD);

    for(int i=0 ; i < NUM ; i++) 
    {
        glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, i,  query[2*i+0]);
        glBeginQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, i,  query[2*i+1]);
    }

    glBeginTransformFeedback(GL_POINTS);  // <-- prim must match output of geometry shader
    glDrawArrays(GL_POINTS, 0, N);
    glEndTransformFeedback();

    for(int i=0 ; i < NUM ; i++) 
    {
        glEndQueryIndexed(GL_PRIMITIVES_GENERATED, i );
        glEndQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, i );
    }
 
    glDisable(GL_RASTERIZER_DISCARD);
 
    GLfloat feedback[N];    
    for(int i=0 ; i < NUM ; i++)
    {
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, tbo[i]); // <-- without thus pullback 2nd buffer twice
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(feedback), feedback);

        std::cout << " feedback " << i << " ( " ; 
        for (int i = 0; i < N ; i++) std::cout << std::setw(10) << feedback[i] << " " ; 
        std::cout << " ) " << std::endl ;         

    }

    for(int i=0 ; i < NUM ; i++)
    {       
        glGetQueryObjectiv(query[2*i+0], GL_QUERY_RESULT, &count[2*i+0] );
        glGetQueryObjectiv(query[2*i+1], GL_QUERY_RESULT, &count[2*i+1] );
    }



/*
    GLint qiv[4] ; 
    glGetQueryiv( GL_PRIMITIVES_GENERATED, GL_CURRENT_QUERY, &qiv[0] );
    glGetQueryiv( GL_PRIMITIVES_GENERATED, GL_QUERY_COUNTER_BITS, &qiv[1] );
    glGetQueryiv( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, GL_CURRENT_QUERY, &qiv[2] );
    glGetQueryiv( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, GL_QUERY_COUNTER_BITS, &qiv[3] );
    
    for(int i=0 ; i < 4 ; i++)
    std::cout << " qiv " << qiv[i] << std::endl ; 
*/ 


    for(int i=0 ; i < NUM ; i++)
    {
         std::cout 
                    << " count[ " << 2*i+0 << "] = " << count[2*i+0]
                    << " count[ " << 2*i+1 << "] = " << count[2*i+1]
                    << std::endl ; 
    }
 



    std::cout << " ///////// WORKAROUND ... REPEATING TransformFeedback against a 1-byte buffer (just for counts) " << std::endl ;

    for(int i=1 ; i < NUM ; i++) glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, tboDEVNULL );

    glEnable(GL_RASTERIZER_DISCARD);

    for(int i=1 ; i < NUM ; i++) 
    {
        //glUniform1i( LOC_SLOT, i );

        glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, i,  query2[2*i+0]);
        glBeginQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, i,  query2[2*i+1]);

        glBeginTransformFeedback(GL_POINTS);  // <-- prim must match output of geometry shader

        glDrawArrays(GL_POINTS, 0, N);

        glEndTransformFeedback();

        glEndQueryIndexed(GL_PRIMITIVES_GENERATED, i );
        glEndQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, i );
    }

    glDisable(GL_RASTERIZER_DISCARD);
 
    for(int i=0 ; i < NUM ; i++)
    {       
        glGetQueryObjectiv(query2[2*i+0], GL_QUERY_RESULT, &count2[2*i+0] );
        glGetQueryObjectiv(query2[2*i+1], GL_QUERY_RESULT, &count2[2*i+1] );
    }

    for(int i=0 ; i < NUM ; i++)
    {
         std::cout 
                    << " count2[ " << 2*i+0 << "] = " << count2[2*i+0]
                    << " count2[ " << 2*i+1 << "] = " << count2[2*i+1]
                    << std::endl ; 
    }
 


    prog->destroy();

    glDeleteBuffers(NUM, tbo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    frame.destroy();

    return 0 ; 
}


