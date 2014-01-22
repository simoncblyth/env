Chroma Materials
=================

Material handling in Chroma
------------------------------

::

    (chroma_env)delta:chroma blyth$ grep -l material *.py
    detector.py   # passing mention, hands parameter to Geometry subclass of Detector
    sim.py        # passing mention, hands detector.detector_material to G4ParallelGenerator
    pmt.py        # pmt Solid construction using material arguments to build funcs
    geometry.py   # implementation of Solid and Material 


::

    (chroma_env)delta:chroma blyth$ find . -name '*.py' -exec grep -l material {} \;
    ./demo/optics.py
    ./demo/pmt.py
    ./detector.py
    ./generator/g4gen.py    # using Material to create G4Material instances
    ./generator/photon.py
    ./geometry.py
    ./gpu/geometry.py
    ./pmt.py
    ./sim.py


chroma/geometry.py
-------------------

Solid
~~~~~~

* constant or iterator inner/outer material parameters

Material
~~~~~~~~~

* constant or over standard wavelengths properties




chroma/cuda/propagate.cu
-------------------------

propagate
~~~~~~~~~~~

Within the propagate stepping the `fill_state(s, p, g)` state, photon, geometry
sets material props with the state.

::

    152     if (p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT))
    153     return;
    ///
    ///     FLAGGED AS A DEAD PHOTON ALREADY, NOTHING TO DO
    ///
    154 
    155     State s;
    156 
    157     int steps = 0;
    158     while (steps < max_steps) {
    159     steps++;
    160 
    161     int command;
    162 
    163     // check for NaN and fail
    164     if (isnan(p.direction.x*p.direction.y*p.direction.z*p.position.x*p.position.y*p.position.z)) {
    165         p.history |= NO_HIT | NAN_ABORT;
    166         break;
    167     }
    168 
    169     fill_state(s, p, g);
    170 
    171     if (p.last_hit_triangle == -1)
    172         break;
    173 
    174     command = propagate_to_boundary(p, s, rng, use_weights, scatter_first);
    175     scatter_first = 0; // Only use the scatter_first value once
    176 
    177     if (command == BREAK)
    178         break;
    179 
    180     if (command == CONTINUE)
    181         continue;
    182 
    183     if (s.surface_index != -1) {
    184       command = propagate_at_surface(p, s, rng, g, use_weights);
    185 
    186         if (command == BREAK)
    187         break;
    188 
    189         if (command == CONTINUE)
    190         continue;
    191     }
    192 
    193     propagate_at_boundary(p, s, rng);
    194 
    195     } // while (steps < max_steps)



chroma/cuda/photon.h 
----------------------

State
~~~~~~~

::

     30 struct State
     31 {
     32     bool inside_to_outside;
     33 
     34     float3 surface_normal;
     35 
     36     float refractive_index1, refractive_index2;
     37     float absorption_length;
     38     float scattering_length;
     39     float reemission_prob;
     40     Material *material1;
     41 
     42     int surface_index;
     43 
     44     float distance_to_boundary;
     45 };



fill_state
~~~~~~~~~~~~~

::

    79 __device__ void
    80 fill_state(State &s, Photon &p, Geometry *g)
    81 {
    82     p.last_hit_triangle = intersect_mesh(p.position, p.direction, g,
    83                                          s.distance_to_boundary,
    84                                          p.last_hit_triangle);
    85 
    86     if (p.last_hit_triangle == -1) {
    87         p.history |= NO_HIT;
    88         return;
    89     }
    90 
    91     Triangle t = get_triangle(g, p.last_hit_triangle);
    92 
    93     unsigned int material_code = g->material_codes[p.last_hit_triangle];
    94 
    95     int inner_material_index = convert(0xFF & (material_code >> 24));
    96     int outer_material_index = convert(0xFF & (material_code >> 16));
    97     s.surface_index = convert(0xFF & (material_code >> 8));
    98 
    99     float3 v01 = t.v1 - t.v0;
    100     float3 v12 = t.v2 - t.v1;
    101 
    102     s.surface_normal = normalize(cross(v01, v12));
    103 
    104     Material *material1, *material2;
    105     if (dot(s.surface_normal,-p.direction) > 0.0f) {
    106         // outside to inside
    107         material1 = g->materials[outer_material_index];
    108         material2 = g->materials[inner_material_index];
    109 
    110         s.inside_to_outside = false;
    111     }
    112     else {
    113         // inside to outside
    114         material1 = g->materials[inner_material_index];
    115         material2 = g->materials[outer_material_index];
    116         s.surface_normal = -s.surface_normal;
    117 
    118         s.inside_to_outside = true;
    119     }
    120 
    121     s.refractive_index1 = interp_property(material1, p.wavelength,
    122                                           material1->refractive_index);
    123     s.refractive_index2 = interp_property(material2, p.wavelength,
    124                                           material2->refractive_index);
    125     s.absorption_length = interp_property(material1, p.wavelength,
    126                                           material1->absorption_length);
    127     s.scattering_length = interp_property(material1, p.wavelength,
    128                                           material1->scattering_length);
    129     s.reemission_prob = interp_property(material1, p.wavelength,
    130                                         material1->reemission_prob);
    131 
    132     s.material1 = material1;
    133 } // fill_state


#. for COLLADA integration need to implement the GDML G4 material (and surface) 
   wavelength array properties into COLLADA extra tags


