// https://gist.github.com/dangets/2926425/download#

#include <math.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


// constants
const unsigned int g_window_width = 512;
const unsigned int g_window_height = 512;

const unsigned int g_mesh_width = 256;
const unsigned int g_mesh_height = 256;


// for a device_vector interoperable with OpenGL, pass ogl_interop_allocator as the allocator type
//typedef thrust::device_vector<float4, thrust::experimental::cuda::ogl_interop_allocator<float4> > gl_vector;
//gl_vector g_vec;
thrust::device_ptr<float4> dev_ptr;
GLuint vbo;
struct cudaGraphicsResource *vbo_cuda;

float g_anim = 0.0;

// mouse controls
int g_mouse_old_x, g_mouse_old_y;
int g_mouse_buttons = 0;
float g_rotate_x = 0.0, g_rotate_y = 0.0;
float g_translate_z = -3.0;


struct sine_wave
{
    sine_wave(unsigned int w, unsigned int h, float t)
        : width(w), height(h), time(t) {}

    __host__ __device__
    float4 operator()(unsigned int i)
    {
        unsigned int x = i % width;
        unsigned int y = i / width;

        // calculate uv coordinates
        float u = x / (float) width;
        float v = y / (float) height;
        u = u*2.0f - 1.0f;
        v = v*2.0f - 1.0f;

        // calculate simple sine wave pattern
        float freq = 4.0f;
        float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

        // write output vertex
        return make_float4(u, w, v, 1.0f);
    }

    float time;
    unsigned int width, height;
};


void display(void)
{
    float4 *raw_ptr;
    size_t buf_size;

    cudaGraphicsMapResources(1, &vbo_cuda, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &buf_size, vbo_cuda);
    dev_ptr = thrust::device_pointer_cast(raw_ptr);

    // transform the mesh
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last(g_mesh_width * g_mesh_height);

    thrust::transform(first, last, dev_ptr,
            sine_wave(g_mesh_width, g_mesh_height, g_anim));

    cudaGraphicsUnmapResources(1, &vbo_cuda, 0);


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, g_translate_z);
    glRotatef(g_rotate_x, 1.0, 0.0, 0.0);
    glRotatef(g_rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, g_mesh_width * g_mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutSwapBuffers();
    glutPostRedisplay();

    g_anim += 0.001;
}


void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        g_mouse_buttons |= 1<<button;
    } else if(state == GLUT_UP) {
        g_mouse_buttons = 0;
    }

    g_mouse_old_x = x;
    g_mouse_old_y = y;

    glutPostRedisplay();
}


void motion(int x, int y)
{
    float dx, dy;
    dx = x - g_mouse_old_x;
    dy = y - g_mouse_old_y;

    if(g_mouse_buttons & 1) {
        g_rotate_x += dy * 0.2;
        g_rotate_y += dx * 0.2;
    } else if (g_mouse_buttons & 4) {
        g_translate_z += dy * 0.01;
    }

    g_mouse_old_x = x;
    g_mouse_old_y = y;
}


void keyboard(unsigned char key, int, int)
{
    switch(key)
    {
        case(27):
            // deallocate memory
            //g_vec.clear();
            //g_vec.shrink_to_fit();
            exit(0);
        default:
            break;
    }
}


int main(int argc, char** argv)
{
    // Create GL context
    glutInit(&argc, argv);

    //glutInitContextVersion(4, 0);
    //glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
    //glutInitContextProfile(GLUT_CORE_PROFILE);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(g_window_width, g_window_height);
    glutCreateWindow("Thrust/GL interop");

    GLenum glewInitResult = glewInit();
    if (glewInitResult != GLEW_OK) {
        throw std::runtime_error("Couldn't initialize GLEW");
    }


    // initialize GL
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, g_window_width, g_window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)g_window_width / (GLfloat)g_window_height, 0.1, 10.0);

    cudaGLSetGLDevice(0);


    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);


    unsigned int size = g_mesh_width * g_mesh_height * sizeof(float4);
    // create vbo
    glGenBuffers(1, &vbo);
    // bind, initialize, unbind
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&vbo_cuda, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // transform the mesh
    //thrust::counting_iterator<int,thrust::device_space_tag> first(0);
    //thrust::counting_iterator<int,thrust::device_space_tag> last(g_mesh_width * g_mesh_height);
    //thrust::transform(first, last, dev_ptr,
    //        sine_wave(g_mesh_width, g_mesh_height, g_anim));

    // start rendering mainloop
    glutMainLoop();

    // TODO: free a bunch of junk


    return 0;
}
