#include <iostream>
#include "Box.hh"
#include "Buf.hh"
#include "UV.hh"

Box::Box(unsigned level_, float extent)
    :
    level(level_),
    bb(extent)
{
    init();
}


void Box::init()
{
    int nu = 1 << level ;
    int nv = 1 << level ;
    for(int s=0 ; s < 6 ; s++)  // sheet
    {
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

                glm::vec4 p00 = par_pos_model(u00);
                glm::vec4 p10 = par_pos_model(u10);
                glm::vec4 p01 = par_pos_model(u01);
                glm::vec4 p11 = par_pos_model(u11);

                tri_vert.push_back(p00) ;
                tri_vert.push_back(p10) ;
                tri_vert.push_back(p11) ;

                tri_vert.push_back(p01) ;
                tri_vert.push_back(p00) ;
                tri_vert.push_back(p11) ;

            }
        }
    }
}

Buf* Box::buf()
{ 
    unsigned num_vert = tri_vert.size() ;
    assert( sizeof(glm::vec4) == sizeof(float)*4 );
    unsigned size = sizeof(glm::vec4)*num_vert ; 
    return new Buf( num_vert, size , (void*)tri_vert.data() ) ; 
}  


glm::vec4 Box::par_pos_model(const UV& uv )
{
    unsigned s = uv.s() ; 
    float fu = uv.fu() ;
    float fv = uv.fv() ;

    glm::vec4 p ; 
    p.w = 1.f ; 

    switch(s)
    {
        case 0:{    // -Z
                  p.x = glm::mix( bb.min.x, bb.max.x, 1 - fu ) ;
                  p.y = glm::mix( bb.min.y, bb.max.y, fv ) ;
                  p.z = bb.min.z ;
               }
               ; break ;
        case 1:{   // +Z
                  p.x = glm::mix( bb.min.x, bb.max.x, fu ) ;
                  p.y = glm::mix( bb.min.y, bb.max.y, fv ) ;
                  p.z = bb.max.z ;
               }
               ; break ;


        case 2:{   // -X
                  p.x = bb.min.x ;
                  p.y = glm::mix( bb.min.y, bb.max.y, 1 - fu ) ;
                  p.z = glm::mix( bb.min.z, bb.max.z, fv ) ;
               }
               ; break ;
        case 3:{   // +X
                  p.x = bb.max.x ;
                  p.y = glm::mix( bb.min.y, bb.max.y, fu ) ;
                  p.z = glm::mix( bb.min.z, bb.max.z, fv ) ;
               }
               ; break ;



        case 4:{  // -Y
                  p.x = glm::mix( bb.min.x, bb.max.x, fu ) ;
                  p.y = bb.min.y ;
                  p.z = glm::mix( bb.min.z, bb.max.z, fv ) ;
               }
               ; break ;
        case 5:{  // +Y
                  p.x = glm::mix( bb.min.x, bb.max.x, 1 - fu ) ;
                  p.y = bb.max.y ;
                  p.z = glm::mix( bb.min.z, bb.max.z, fv ) ;
               }
               ; break ;
    }


    std::cout << "box::par_pos_model"
              << " uv " << uv.desc()
       //       << " p " << glm::to_string(p)
              << std::endl 
               ; 

    return p ;
}


