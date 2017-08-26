/*
https://open.gl/feedback
https://open.gl/content/code/c8_geometry.txt

*/

// Link statically with GLEW
#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cassert>


static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

struct Frame 
{
   GLFWwindow* window ;
   Frame() : window(NULL) 
   {
       init();
   }
   void init();
   void destroy();
};


void Frame::init()
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3); 
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2); 
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);


    // Initialize GLEW
    glewExperimental = GL_TRUE;
    glewInit();


   // get version info
    const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString (GL_VERSION); // version as a string
    std::cout << "Frame::gl_init_window Renderer: " << renderer << std::endl ; 
    std::cout << "Frame::gl_init_window OpenGL version supported " <<  version << std::endl ;
}


void Frame::destroy()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}



struct Prog
{
    static const GLchar* vertexShaderSrc ;
    static const GLchar* geoShaderSrc ;

    GLuint program ; 
    GLuint vertexShader ;
    GLuint geoShader ;

    Prog()
    {
        vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSrc, nullptr);
        glCompileShader(vertexShader);

        geoShader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geoShader, 1, &geoShaderSrc, nullptr);
        glCompileShader(geoShader);

        program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, geoShader);

        // after program creation but before linking 
        // associate attributes to capture into buffer via transform feedback

        const GLchar* feedbackVaryings[] = { "outValue" };
        glTransformFeedbackVaryings(program, 1, feedbackVaryings, GL_INTERLEAVED_ATTRIBS);

        glLinkProgram(program);
        glUseProgram(program);
    }

    void destroy()
    {
        glDeleteProgram(program);
        glDeleteShader(geoShader);
        glDeleteShader(vertexShader);
    }

    GLint getAttribLocation(const char* att)
    {
        return glGetAttribLocation(program, att );
    }
};



//  C++11 raw string literal
const GLchar* Prog::vertexShaderSrc = R"glsl(

    #version 400 core
    in float inValue;
    out float geoValue;
    void main()
    {
        geoValue = sqrt(inValue);
    }

)glsl";

// Geometry shader
const GLchar* Prog::geoShaderSrc = R"glsl(

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



int main(void)
{
    Frame frame ; 
    Prog prog ; 

    // Create VAO
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create input VBO and vertex format

    static const unsigned N = 5 ; 
    GLfloat data[N] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

    // desc the input data
    GLint inputAttrib = prog.getAttribLocation("inValue");
    glEnableVertexAttribArray(inputAttrib);
    glVertexAttribPointer(inputAttrib, 1, GL_FLOAT, GL_FALSE, 0, 0);

    // Create transform feedback buffer, 3 times larger from geo amplification
    GLuint tbo;
    glGenBuffers(1, &tbo);
    glBindBuffer(GL_ARRAY_BUFFER, tbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), nullptr, GL_STATIC_READ);


    GLuint query;
    glGenQueries(1, &query);

    // both these give 5
    //GLenum qtarget = GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN ;
    GLenum qtarget = GL_PRIMITIVES_GENERATED ;


    // how does GL know to target output into the tbo ?  from below glBindBufferBase

    // Perform feedback transform
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


    prog.destroy();

    glDeleteBuffers(1, &tbo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    frame.destroy();

    exit(EXIT_SUCCESS);
}


