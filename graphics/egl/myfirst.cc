#include <EGL/egl.h>
#include <GL/gl.h>
#include <stdio.h>
#include <iostream>


/**
https://www.khronos.org/registry/EGL/sdk/docs/man/html/eglChooseConfig.xhtml

EGL_SURFACE_TYPE

Must be followed by a bitmask indicating which EGL surface types and
capabilities the frame buffer configuration must support. 
Mask bits include::

    EGL_PBUFFER_BIT
    Config supports creating pixel buffer surfaces.

    ...

    EGL_WINDOW_BIT
    Config supports creating window surfaces.

For example, if the bitmask is set to EGL_WINDOW_BIT | EGL_PIXMAP_BIT, only
frame buffer configurations that support both windows and pixmaps will be
considered. The default value is EGL_WINDOW_BIT.

**/


static const EGLint configAttribs[] = {
    EGL_SURFACE_TYPE,  EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE
};    

static const int pbufferWidth = 300;
static const int pbufferHeight = 300;

static const EGLint pbufferAttribs[] = {
    EGL_WIDTH, pbufferWidth,
    EGL_HEIGHT, pbufferHeight,
    EGL_NONE,
};

void saveppm(const char* filename, int width, int height, unsigned char* image) {
    // save into ppm
    FILE * fp;
    fp = fopen(filename, "wb");

    int ncomp = 4;
    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned char* data = new unsigned char[height*width*3] ; 

    for( int y=height-1; y >= 0; --y ) // flip vertically
        {   
            for( int x=0; x < width ; ++x ) 
                {   
                    *(data + (y*width+x)*3+0) = image[(y*width+x)*ncomp+0] ;   
                    *(data + (y*width+x)*3+1) = image[(y*width+x)*ncomp+1] ;   
                    *(data + (y*width+x)*3+2) = image[(y*width+x)*ncomp+2] ;   
                }
        } 
    fwrite(data, sizeof(unsigned char)*height*width*3, 1, fp);
    fclose(fp);  

    delete[] data;
}




static void err_check(const char* msg)
{
    EGLint er = eglGetError() ;
    if( er != EGL_SUCCESS )
    std::cout << msg    
              << " err 0x" << std::hex << er << std::dec << std::endl ;          
}

static void dump(const char* msg, int val)
{
    std::cout << msg    
              << "  0x" << std::hex << val << std::dec << std::endl ;          
}


/**
https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/


**/

int main(int argc, char** argv)
{
    // 1. Initialize EGL
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    std::cout << " display " << display <<  std::endl ; 

    //dump("display", display );
    

    err_check("eglGetDisplay...");

    if (display == EGL_NO_DISPLAY) {
        fprintf(stderr, "failed to eglGetDisplay\n");
        return false;
    }

    printf(" EGL_BAD_DISPLAY %d \n", EGL_BAD_DISPLAY  ); 


    EGLint major, minor;

    eglInitialize(display, &major, &minor);
    err_check("eglInitialize...");
    std::cout << "egl error " << eglGetError() << std::endl;
    std::cout << "major/minor: " << major << "/" << minor << std::endl;

    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;

    eglChooseConfig(display, configAttribs, &eglCfg, 1, &numConfigs);
    std::cout << "egl error " << eglGetError() << std::endl;
    std::cout << "numConfigs: " << numConfigs << std::endl;

    // 3. Create a surface
    EGLSurface surface = eglCreatePbufferSurface(display, eglCfg, 
                                                 pbufferAttribs);

    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);

    // 5. Create a context and make it current
    EGLContext eglCtx = eglCreateContext(display, eglCfg, EGL_NO_CONTEXT, 
                                         NULL);

    eglMakeCurrent(display, surface, surface, eglCtx);

    // from now on use your OpenGL context
    GLuint renderBufferWidth = pbufferWidth;
    GLuint renderBufferHeight = pbufferHeight;

    int size = 4 * renderBufferHeight * renderBufferWidth;
    unsigned char *data2 = new unsigned char[size];

    float red ; 

    char filename[255];
    for (int i = 0; i < 100; ++i) {
        // std::cout << "HELLO: " << i << std::endl;

        red = 0.5f*i/100 ;

        printf("%d %f \n", i, red );

        glClearColor(red, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glFlush();


        eglSwapBuffers( display, surface);
    
        glPixelStorei(GL_PACK_ALIGNMENT,1); /* byte aligned output */
        glReadPixels(0,0,renderBufferWidth,renderBufferHeight,GL_RGBA, GL_UNSIGNED_BYTE, data2);
        snprintf(filename, 255, "fig-myfirst%02d.ppm", i);
        saveppm(filename, renderBufferWidth, renderBufferHeight, data2);
    }
    // 6. Terminate EGL when finished
    eglTerminate(display);
    return 0;
}
