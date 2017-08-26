#include "Sphere.hh"
#include "UV.hh"

Sphere::Sphere(int level_, float radius_ , float cx, float cy, float cz) 
    :
    Prim(),
    level(level_),
    radius(radius_),
    center(cx,cy,cz)
{
    init(); 
}

void Sphere::init()
{

    int nu = 1 << level ;
    int nv = 1 << level ;
    int s = 0 ; 

    for (int v = 0; v < nv ; v++)
    {
    for (int u = 0 ; u < nu ; u++)
    {

/*

     p01     p11
       +-----+
       |   / |
       | /   |
       +-----+
     p00    p10
  
*/

            UV u00 = make_UV(s,u+0,v+0,nu,nv, 0);
            UV u10 = make_UV(s,u+1,v+0,nu,nv, 0);
            UV u01 = make_UV(s,u+0,v+1,nu,nv, 0);
            UV u11 = make_UV(s,u+1,v+1,nu,nv, 0);

            glm::vec3 p00 = par_pos_model(u00);
            glm::vec3 p10 = par_pos_model(u10);
            glm::vec3 p01 = par_pos_model(u01);
            glm::vec3 p11 = par_pos_model(u11);

            unsigned v00 = add_vert(p00);
            unsigned v10 = add_vert(p10);
            unsigned v01 = add_vert(p01);
            unsigned v11 = add_vert(p11);

            add_quad( v00, v10, v11, v01 );
    }
    }
    
    populate();
}

glm::vec3 Sphere::par_pos_model(const UV& uv) const 
{
    unsigned s  = uv.s(); 
    assert(s == 0); 

    glm::vec3 pos(center);
    _par_pos_body( pos, uv, radius );
  
    return pos ; 
}

void Sphere::_par_pos_body(glm::vec3& pos,  const UV& uv, const float r_ )  // static
{
    unsigned  v  = uv.v(); 
    unsigned nv  = uv.nv(); 

    // Avoid numerical precision problems at the poles
    // by providing precisely the same positions
    // and on the 360 degree seam by using 0 degrees at 360 
    
    bool is_north_pole = v == 0 ; 
    bool is_south_pole = v == nv ; 

    if(is_north_pole || is_south_pole) 
    {   
        pos += glm::vec3(0,0,is_north_pole ? r_ : -r_ ) ; 
    }   
    else
    {   
        bool seamed = true ; 
        float azimuth = uv.fu2pi(seamed); 
        float polar = uv.fvpi() ; 
        float ca = cosf(azimuth);
        float sa = sinf(azimuth);
        float cp = cosf(polar);
        float sp = sinf(polar);

        pos += glm::vec3( r_*ca*sp, r_*sa*sp, r_*cp );
    }   
}


