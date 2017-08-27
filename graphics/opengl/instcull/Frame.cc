#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cassert>


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLEQ_IMPLEMENTATION
#include "GLEQ.hh"



#include "Att.hh"
#include "Frame.hh"




static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}

/*
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}
*/

Frame::Frame(const char* title_, int width, int height)  
    :
    title(title_),
    window(NULL)
{
    init(title, width, height);
} 

float Frame::updateWindowTitle(const char* status)
{
    static double previous_seconds = glfwGetTime (); 
    static int frame_count;
    double current_seconds = glfwGetTime (); 
    double elapsed_seconds = current_seconds - previous_seconds;
    if (elapsed_seconds > 0.25) 
    {
        previous_seconds = current_seconds;
        double fps = (double)frame_count / elapsed_seconds;
        char tmp[128];
        sprintf (tmp, "%s %s fps: %.2f ", title, status ? status : "-" , fps );
        glfwSetWindowTitle (window, tmp);
        frame_count = 0;
    }
    frame_count++;
    return (float)current_seconds ; 
}


void Frame::init(const char* title, int width, int height)
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

    window = glfwCreateWindow(width, height,  title, NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);

    gleqTrackWindow(window);  // hookup the callbacks and arranges outcomes into event queue 


    // Initialize GLEW
    glewExperimental = GL_TRUE;
    glewInit();


   // get version info
    const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString (GL_VERSION); // version as a string
    std::cout << "Frame::gl_init_window Renderer: " << renderer << std::endl ; 
    std::cout << "Frame::gl_init_window OpenGL version supported " <<  version << std::endl ;


    Q qq0("GL_MAX_TEXTURE_BUFFER_SIZE(texels)", GL_MAX_TEXTURE_BUFFER_SIZE);
    Q qq1("GL_MAX_UNIFORM_BLOCK_SIZE", GL_MAX_UNIFORM_BLOCK_SIZE);

    Q qq2("GL_MAX_VERTEX_STREAMS", GL_MAX_VERTEX_STREAMS );
    Q qq3("GL_QUERY_COUNTER_BITS", GL_QUERY_COUNTER_BITS );
    Q qq4("GL_MAX_TRANSFORM_FEEDBACK_BUFFERS", GL_MAX_TRANSFORM_FEEDBACK_BUFFERS );
    Q qq5("GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS", GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS );


}


void Frame::listen()
{
    glfwPollEvents();
    GLEQevent event;
    while (gleqNextEvent(&event))
    {   
        handle_event(event);
        gleqFreeEvent(&event);
    }   
}

void Frame::handle_event(GLEQevent& event)
{
    std::cerr <<"Frame::handle_event" << std::endl ;
    switch (event.type)
    {   
        case GLEQ_KEY_PRESSED:
             key_pressed(event.key.key);
             break;
        case GLEQ_KEY_RELEASED:
             key_released(event.key.key);
             break;
        default:
             std::cerr <<"Frame::handle_event (other)" << std::endl ;
             break; 
    }   
}

void Frame::key_pressed(unsigned key)
{
    std::cerr <<"Frame::key_pressed" << std::endl ;
    if( key == GLFW_KEY_ESCAPE)
    {   
        std::cerr <<"Frame::key_pressed escape" << std::endl ;
        glfwSetWindowShouldClose (window, 1); 
    }
    else
    {
        if(key < NUM_KEYS) keys_down[key] = true ; 
        std::cerr <<"Frame::key_pressed " <<  key << std::endl ;
    }
}

void Frame::key_released(unsigned key)
{
    if(key < NUM_KEYS) keys_down[key] = false ; 
}


void Frame::destroy()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}


