#include <string>
#include <iostream>

//#define YGLTF_INLINE 1
//#define YGLTF_NO_IMAGE 1

#include "yocto_gltf.h"







/*
Using extracts from
    /usr/local/env/graphics/yoctogl/yocto-gl/yocto/yocto_gltf.cpp 

*/



//
// Math support
//
static inline std::array<float, 16> _float4x4_mul(
    const std::array<float, 16>& a, const std::array<float, 16>& b) {
    auto c = std::array<float, 16>();
    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 4; j++) {
            c[j * 4 + i] = 0;
            for (auto k = 0; k < 4; k++)
                c[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
        }
    }
    return c;
}




//
// Identity matrix
//
const std::array<float, 16> _identity_float4x4 = {
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};



void test_walk(const ygltf::glTF_t* gltf, int scene_idx=-1)
{
    std::cout << "test_walk" 
              << " scene_idx " << scene_idx 
              << std::endl ; 

    // get scene names
    auto scenes = std::vector<int>();
    if (scene_idx < 0) {
        for (auto i = 0; i < gltf->scenes.size(); i++) scenes.push_back(i);
    } else {
        scenes.push_back(scene_idx);
    }

    // walk the scenes and add objects
    int count = 0 ; 

    for (auto scn_id : scenes) 
    {
        auto scn = &gltf->scenes.at(scn_id);

        // initialize stack of node transforms to identity matrix
        auto stack = std::vector<std::tuple<int, std::array<float, 16>>>();
        for (auto node_id : scn->nodes) 
        {
            stack.push_back(std::make_tuple(node_id, _identity_float4x4));
        }

        while (!stack.empty()) 
        {
            int              node_id;
            std::array<float, 16> xf;
            std::tie(node_id, xf)   = stack.back();
            stack.pop_back();   

            // popping from the back,  hmm does the root node need to last ?

            auto node = &gltf->nodes.at(node_id);

            xf = _float4x4_mul(xf, node_transform(node));   //   T-R-S-M    

            if( node->mesh == 3)
            { 

                count++ ;  
                std::cout << " node.id " << node_id << " node.mesh " << node->mesh << " node.name:" << node->name << std::endl ; 
            
                std::cout << "lxf:" ; 
                for(const auto& s: node->matrix ) std::cout << s << ' ';
                std::cout << std::endl ; 

                std::cout << "gxf: " ; 
                for(const auto& s: xf ) std::cout << s << ' ';
                std::cout << std::endl ; 
            }


            for (auto child : node->children) { stack.push_back({child, xf}); }
        }
    }

    std::cout << "count:" << count << std::endl; 
}



int main(int argc, char** argv)
{
    const std::string filename = argv[1] ; 
    std::cout << "load_gltf " << filename << std::endl ; 


    bool load_bin = true ; 
    bool load_shaders = true ; 
    bool load_img = false ; 
    bool skip_missing = true  ; 

    auto gltf = std::unique_ptr<ygltf::glTF_t>(ygltf::load_gltf(filename, load_bin, load_shaders, load_img, skip_missing )) ;

    test_walk(gltf.get(), gltf->scene) ;


    return 0 ; 
}
