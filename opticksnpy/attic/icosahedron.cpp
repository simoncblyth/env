//  https://www.cosc.brocku.ca/Offerings/3P98/course/OpenGL/glut-3.7/progs/advanced/sphere.c
/* sphere.c - by David Blythe, SGI */

/* Instead of tessellating a sphere by lines of longitude and latitude
   (a technique that over tessellates the poles and under tessellates
   the equator of the sphere), tesselate based on regular solids for a
   more uniform tesselation.

   This approach is arguably better than the gluSphere routine's
   approach using slices and stacks (latitude and longitude). -mjk */


#include "icosahedron.hpp"

#include <cstring>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float x, y, z;
} point;

struct triangle 
{
   void copyTo(float* b)
   {
       memcpy(b+0, &pt[0], sizeof(float)*3);
       memcpy(b+3, &pt[1], sizeof(float)*3);
       memcpy(b+6, &pt[2], sizeof(float)*3);
   }
   point pt[3];
};



/* six equidistant points lying on the unit sphere */
#define XPLUS {  1,  0,  0 }    /*  X */
#define XMIN  { -1,  0,  0 }    /* -X */
#define YPLUS {  0,  1,  0 }    /*  Y */
#define YMIN  {  0, -1,  0 }    /* -Y */
#define ZPLUS {  0,  0,  1 }    /*  Z */
#define ZMIN  {  0,  0, -1 }    /* -Z */


#define PX   { 1, 0,  0} 
#define MX   {-1, 0,  0}
#define PY   { 0, 1,  0} 
#define MY   { 0,-1,  0}
#define PZ   { 0, 0,  1} 
#define MZ   { 0, 0, -1}


static triangle _octahedron[8] = {
        /* top pyramid */
        { {PX,  PZ,  MY}, },
        { {MY,  PZ,  MX}, },
        { {MX,  PZ,  PY}, },
        { {PY,  PZ,  PX}, },
        /* bottom pyramid */
        { {PX,  MY,  MZ}, },
        { {MY,  MX,  MZ}, },
        { {MX,  PY,  MZ}, },
        { {PY,  PX,  MZ}, }
};

float* octahedron_()
{
    float* buf = (float *)malloc(8*3*3*sizeof(float));
    for(int s = 0; s < 8; s++) 
    {
        triangle *t = &_octahedron[s];
        t->copyTo(buf+s*3*3); 
    }
    return buf ;
}



/* for icosahedron */
#define CZ (0.89442719099991)   /*  2/sqrt(5) */
#define SZ (0.44721359549995)   /*  1/sqrt(5) */
#define C1 (0.951056516)        /* cos(18),  */
#define S1 (0.309016994)        /* sin(18) */
#define C2 (0.587785252)        /* cos(54),  */
#define S2 (0.809016994)        /* sin(54) */
#define X1 (C1*CZ)
#define Y1 (S1*CZ)
#define X2 (C2*CZ)
#define Y2 (S2*CZ)

#define Ip0     {0.,    0.,     1.}
#define Ip1     {-X2,   -Y2,    SZ}
#define Ip2     {X2,    -Y2,    SZ}
#define Ip3     {X1,    Y1,     SZ}
#define Ip4     {0,     CZ,     SZ}
#define Ip5     {-X1,   Y1,     SZ}

#define Im0     {-X1,   -Y1,    -SZ}
#define Im1     {0,     -CZ,    -SZ}
#define Im2     {X1,    -Y1,    -SZ}
#define Im3     {X2,    Y2,     -SZ}
#define Im4     {-X2,   Y2,     -SZ}
#define Im5     {0.,    0.,     -1.}

/* vertices of a unit icosahedron */
static triangle _icosahedron[20]= {
        /* front pole */
        { {Ip0, Ip1, Ip2}, },
        { {Ip0, Ip5, Ip1}, },
        { {Ip0, Ip4, Ip5}, },
        { {Ip0, Ip3, Ip4}, },
        { {Ip0, Ip2, Ip3}, },

        /* mid */
        { {Ip1, Im0, Im1}, },
        { {Im0, Ip1, Ip5}, },
        { {Ip5, Im4, Im0}, },
        { {Im4, Ip5, Ip4}, },
        { {Ip4, Im3, Im4}, },
        { {Im3, Ip4, Ip3}, },
        { {Ip3, Im2, Im3}, },
        { {Im2, Ip3, Ip2}, },
        { {Ip2, Im1, Im2}, },
        { {Im1, Ip2, Ip1}, },

        /* back pole */
        { {Im3, Im2, Im5}, },
        { {Im4, Im3, Im5}, },
        { {Im0, Im4, Im5}, },
        { {Im1, Im0, Im5}, },
        { {Im2, Im1, Im5}, },
};

float* icosahedron_()
{
    float* buf = (float *)malloc(20*3*3*sizeof(float));
    for(int s = 0; s < 20; s++) 
    {
        triangle *t = &_icosahedron[s];
        t->copyTo(buf+s*3*3); 
    }
    return buf ;
}




/* normalize point r */
static void
normalize(point *r) {
    float mag;

    mag = r->x * r->x + r->y * r->y + r->z * r->z;
    if (mag != 0.0f) {
        mag = 1.0f / sqrt(mag);
        r->x *= mag;
        r->y *= mag;
        r->z *= mag;
    }
}

/* linearly interpolate between a & b, by fraction f */
static void
lerp(point *a, point *b, float f, point *r) {
    r->x = a->x + f*(b->x-a->x);
    r->y = a->y + f*(b->y-a->y);
    r->z = a->z + f*(b->z-a->z);
}



static void 
V(float* b, point* a, point* c, point* v)
{
   point x  ; 
   x.x = v->x ; 
   x.y = v->y ; 
   x.z = v->z ; 

   normalize(a);
   normalize(c);
   normalize(&x);

   b[0] = a->x; b[1] = a->y; b[2] = a->z; 
   b[3] = c->x; b[4] = c->y; b[5] = c->z; 
   b[6] = x.x;  b[7] = x.y;  b[8] = x.z; 

}



int icosahedron_ntris(int maxlevel)
{
    int n = 20*(1 << (maxlevel * 2));
    return n ; 
}



float* icosahedron_tris(int maxlevel) 
{
    int nrows = 1 << maxlevel;
    int s, n;
    float *buf, *b;

    n = 20*(1 << (maxlevel * 2));
    b = buf = (float *)malloc(n*3*3*sizeof(float));

    /* iterate over the 20 sides of the icosahedron */
    for(s = 0; s < 20; s++) {
        int i;
        triangle *t = &_icosahedron[s];
        for(i = 0; i < nrows; i++) {
            /* create a tstrip for each row */
            /* number of triangles in this row is number in previous +2 */
            /* strip the ith trapezoid block */
            point v0, v1, v2, v3, va, vb, x1, x2;
            int j;
            lerp(&t->pt[1], &t->pt[0], (float)(i+1)/nrows, &v0);
            lerp(&t->pt[1], &t->pt[0], (float)i/nrows, &v1);
            lerp(&t->pt[1], &t->pt[2], (float)(i+1)/nrows, &v2);
            lerp(&t->pt[1], &t->pt[2], (float)i/nrows, &v3);

            x1 = v0;
            x2 = v1;
            for(j = 0; j < i; j++) {
                /* calculate 2 more vertices at a time */
                lerp(&v0, &v2, (float)(j+1)/(i+1), &va);
                lerp(&v1, &v3, (float)(j+1)/i, &vb);

                V(b,&x1,&x2,&va); x1 = x2; x2 = va;
                b+=9; 

                V(b,&vb,&x2,&x1); x1 = x2; x2 = vb;
                b+=9; 
                
            }
            V(b, &x1, &x2, &v2);
            b+=9; 

        }
    }
    return buf;
}



