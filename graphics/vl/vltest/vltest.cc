#include <GLFW/glfw3.h>
#include <vlCore/VisualizationLibrary.hpp>
#include "Applets/App_RotatingCube.hpp"
#include <stdio.h>

using namespace vl;

int main ( int argc, char *argv[] )
{
  if (!glfwInit ()) {
    fprintf (stderr, "ERROR: could not start GLFW3\n");
    return 1;
  }   

#ifdef  __APPLE__
  glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3); 
  glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2); 
  glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

  // get version info
  const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
  const GLubyte* version = glGetString (GL_VERSION); // version as a string
  printf ("Renderer: %s\n", renderer);
  printf ("OpenGL version supported %s\n", version);

  glEnable (GL_DEPTH_TEST); 

  GLFWwindow* window = glfwCreateWindow (640, 480, "Hello Triangle", NULL, NULL);
  if (!window) {
    fprintf (stderr, "ERROR: could not open window with GLFW3\n");
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent (window);
    


  VisualizationLibrary::init();
  OpenGLContextFormat format;
  format.setDoubleBuffer(true);
  format.setRGBABits( 8,8,8,8 );
  format.setDepthBufferBits(24);
  format.setStencilBufferBits(8);
  format.setFullscreen(false);

  ref<Applet> applet = new App_RotatingCube;
  applet->initialize();


/*
  ref<vlGLUT::GLUTWindow> glut_window = new vlGLUT::GLUTWindow;
  glut_window->addEventListener(applet.get());
  applet->rendering()->as<Rendering>()->renderer()->setFramebuffer( glut_window->framebuffer() );
  applet->rendering()->as<Rendering>()->camera()->viewport()->setClearColor( black );
  vec3 eye    = vec3(0,10,35); 
  vec3 center = vec3(0,0,0);   
  vec3 up     = vec3(0,1,0); 
  mat4 view_mat = mat4::getLookAt(eye, center, up);
  applet->rendering()->as<Rendering>()->camera()->setViewMatrix( view_mat );

*/



  glfwTerminate();
  return 0;
}

