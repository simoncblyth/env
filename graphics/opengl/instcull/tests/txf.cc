
#include <vector>
#include <iostream>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "Frame.hh"
#include "Prog.hh"

const unsigned LOC_inValue = 0 ;  

const GLchar* vertSrc = R"glsl(

    #version 400 core

    layout ( location = 0 ) in float inValue;
    out float geoValue;
    void main()
    {
        geoValue = sqrt(inValue);
    }

)glsl";

const GLchar* geomSrc = R"glsl(

    #version 400 core

    layout(points) in;
    layout(points, max_vertices = 1) out;

    in float[] geoValue;
    out float outValue;

    void main()
    {
      
        outValue = geoValue[0] ;
        EmitVertex();
        EndPrimitive();
    }

)glsl";



int main(int, char** argv)
{
    Frame frame(argv[0]) ; 

    Prog* prog = new Prog(vertSrc, geomSrc, NULL );

    prog->compile();
    prog->create();

    const GLchar* feedbackVaryings[] = { "outValue" };
    glTransformFeedbackVaryings(prog->program, 1, feedbackVaryings, GL_INTERLEAVED_ATTRIBS);

    glBindAttribLocation(prog->program, LOC_inValue , "inValue");

    prog->link();

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    static const unsigned N = 5 ; 
    GLfloat data[N] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(LOC_inValue);
    glVertexAttribPointer(LOC_inValue, 1, GL_FLOAT, GL_FALSE, 0, 0);

    GLuint tbo;
    glGenBuffers(1, &tbo);
    glBindBuffer(GL_ARRAY_BUFFER, tbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), nullptr, GL_STATIC_READ);

    GLuint query;
    glGenQueries(1, &query);

    // both these give 5
    //GLenum qtarget = GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN ;
    GLenum qtarget = GL_PRIMITIVES_GENERATED ;

    {
        glEnable(GL_RASTERIZER_DISCARD);

        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo); // <-- direct where to put transform feedback output 

        glBeginQuery(qtarget, query);

        glBeginTransformFeedback(GL_POINTS);  // <-- prim must match output of geometry shader
            glDrawArrays(GL_POINTS, 0, N);
        glEndTransformFeedback();

        glEndQuery(qtarget);  // <-- without this get 0 from query 

        glDisable(GL_RASTERIZER_DISCARD);

        glFlush();
    }


    GLuint primitives;
    glGetQueryObjectuiv(query, GL_QUERY_RESULT, &primitives);

    std::cout << " GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN: " << primitives << std::endl ; 
    assert( primitives == 5 );


    // Fetch and print results
    GLfloat feedback[N];
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(feedback), feedback);

    for (int i = 0; i < N ; i++) {
        printf("%f\n", feedback[i]);
    }


    prog->destroy();

    glDeleteBuffers(1, &tbo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    frame.destroy();

    return 0 ; 
}


