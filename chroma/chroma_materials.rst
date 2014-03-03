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
    ./generator/g4gen.py    # using Material to create G4Material instances, for photon generation within the material
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


::

    015 # all material/surface properties are interpolated at these
    016 # wavelengths when they are sent to the gpu
    017 standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)
    ...
    200 class Material(object):
    201     """Material optical properties."""
    202     def __init__(self, name='none'):
    203         self.name = name
    204 
    205         self.refractive_index = None
    206         self.absorption_length = None
    207         self.scattering_length = None
    208         self.set('reemission_prob', 0)
    209         self.set('reemission_cdf', 0)
    210         self.density = 0.0 # g/cm^3
    211         self.composition = {} # by mass
    212 
    213     def set(self, name, value, wavelengths=standard_wavelengths):
    214         if np.iterable(value):
    215             if len(value) != len(wavelengths):
    216                 raise ValueError('shape mismatch')
    217         else:
    218             value = np.tile(value, len(wavelengths))
    219 
    220         self.__dict__[name] = np.array(zip(wavelengths, value), dtype=np.float32)
    221 
    222 # Empty material
    223 vacuum = Material('vacuum')
    224 vacuum.set('refractive_index', 1.0)
    225 vacuum.set('absorption_length', 1e6)
    226 vacuum.set('scattering_length', 1e6)


Surface
~~~~~~~~~

Wavelength dependent surface properties handled similarly::

    229 class Surface(object):
    230     """Surface optical properties."""
    231     def __init__(self, name='none', model=0):
    232         self.name = name
    233         self.model = model
    234 
    235         self.set('detect', 0)
    236         self.set('absorb', 0)
    237         self.set('reemit', 0)
    238         self.set('reflect_diffuse', 0)
    239         self.set('reflect_specular', 0)
    240         self.set('eta', 0)
    241         self.set('k', 0)
    242         self.set('reemission_cdf', 0)
    243 
    244         self.thickness = 0.0
    245         self.transmissive = 0


chroma/gpu/geometry.py
-----------------------

Material properties passed over to GPU side

#. refractive_index
#. absorption_length
#. scattering_length
#. reemission_prob
#. reemission_cdf


Looks like (from chroma/sim.py) have just one GPUGeometry.


::

     13 class GPUGeometry(object):
     14     def __init__(self, geometry, wavelengths=None, print_usage=False, min_free_gpu_mem=300e6):
     15         if wavelengths is None:
     16             wavelengths = standard_wavelengths
     17 
     18         try:
     19             wavelength_step = np.unique(np.diff(wavelengths)).item()
     20         except ValueError:
     21             raise ValueError('wavelengths must be equally spaced apart.')
     22 
     23         geometry_source = get_cu_source('geometry_types.h')
     24         material_struct_size = characterize.sizeof('Material', geometry_source)
     25         surface_struct_size = characterize.sizeof('Surface', geometry_source)
     26         geometry_struct_size = characterize.sizeof('Geometry', geometry_source)
     27 
     28         self.material_data = []
     29         self.material_ptrs = []
     30 
     31         def interp_material_property(wavelengths, property):
     32             # note that it is essential that the material properties be
     33             # interpolated linearly. this fact is used in the propagation
     34             # code to guarantee that probabilities still sum to one.
     35             return np.interp(wavelengths, property[:,0], property[:,1]).astype(np.float32)
     36 
     37         for i in range(len(geometry.unique_materials)):
     38             material = geometry.unique_materials[i]
     39 
     40             if material is None:
     41                 raise Exception('one or more triangles is missing a material.')
     42 
     43             refractive_index = interp_material_property(wavelengths, material.refractive_index)
     44             refractive_index_gpu = ga.to_gpu(refractive_index)
     45             absorption_length = interp_material_property(wavelengths, material.absorption_length)
     46             absorption_length_gpu = ga.to_gpu(absorption_length)
     47             scattering_length = interp_material_property(wavelengths, material.scattering_length)
     48             scattering_length_gpu = ga.to_gpu(scattering_length)
     49             reemission_prob = interp_material_property(wavelengths, material.reemission_prob)
     50             reemission_prob_gpu = ga.to_gpu(reemission_prob)
     51             reemission_cdf = interp_material_property(wavelengths, material.reemission_cdf)
     52             reemission_cdf_gpu = ga.to_gpu(reemission_cdf)
     ..

NB all five material properties are linearly interpolated onto the same standard (or provided) wavelengths. 


Same class also passes surface properties

#. detect
#. absorb
#. reemit
#. reflect_diffuse
#. reflect_specular
#. eta
#. k 
#. reemission_cdf


::

     78         for i in range(len(geometry.unique_surfaces)):
     79             surface = geometry.unique_surfaces[i]
     80 
     81             if surface is None:
     82                 # need something to copy to the surface array struct
     83                 # that is the same size as a 64-bit pointer.
     84                 # this pointer will never be used by the simulation.
     85                 self.surface_ptrs.append(np.uint64(0))
     86                 continue
     87 
     88             detect = interp_material_property(wavelengths, surface.detect)
     89             detect_gpu = ga.to_gpu(detect)
     90             absorb = interp_material_property(wavelengths, surface.absorb)
     91             absorb_gpu = ga.to_gpu(absorb)
     92             reemit = interp_material_property(wavelengths, surface.reemit)
     93             reemit_gpu = ga.to_gpu(reemit)
     94             reflect_diffuse = interp_material_property(wavelengths, surface.reflect_diffuse)
     95             reflect_diffuse_gpu = ga.to_gpu(reflect_diffuse)
     96             reflect_specular = interp_material_property(wavelengths, surface.reflect_specular)
     97             reflect_specular_gpu = ga.to_gpu(reflect_specular)
     98             eta = interp_material_property(wavelengths, surface.eta)
     99             eta_gpu = ga.to_gpu(eta)
     100             k = interp_material_property(wavelengths, surface.k)
     101             k_gpu = ga.to_gpu(k)
     102             reemission_cdf = interp_material_property(wavelengths, surface.reemission_cdf)
     103             reemission_cdf_gpu = ga.to_gpu(reemission_cdf)
     104 
     105             self.surface_data.append(detect_gpu)
     106             self.surface_data.append(absorb_gpu)
     107             self.surface_data.append(reemit_gpu)
     108             self.surface_data.append(reflect_diffuse_gpu)
     109             self.surface_data.append(reflect_specular_gpu)
     110             self.surface_data.append(eta_gpu)
     111             self.surface_data.append(k_gpu)
     112             self.surface_data.append(reemission_cdf_gpu)
     113 
     114             surface_gpu = \
     115                 make_gpu_struct(surface_struct_size,
     116                                 [detect_gpu, absorb_gpu, reemit_gpu,
     117                                  reflect_diffuse_gpu,reflect_specular_gpu,
     118                                  eta_gpu, k_gpu, reemission_cdf_gpu,
     119                                  np.uint32(surface.model),
     120                                  np.uint32(len(wavelengths)),
     121                                  np.uint32(surface.transmissive),
     122                                  np.float32(wavelength_step),
     123                                  np.float32(wavelengths[0]),
     124                                  np.float32(surface.thickness)])
     125 
     126             self.surface_ptrs.append(surface_gpu)




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




material_codes
~~~~~~~~~~~~~~~~~


packed in `chroma/geometry.py` 4 bytes, high to low: `material1/material2/surface/blank` ::

       material_codes = (((geometry.material1_index & 0xff) << 24) |
                         ((geometry.material2_index & 0xff) << 16) |
                         ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)
       self.material_codes = ga.to_gpu(material_codes)


unpacked::

    92 
    93     unsigned int material_code = g->material_codes[p.last_hit_triangle];
    94 
    95     int inner_material_index = convert(0xFF & (material_code >> 24));
    96     int outer_material_index = convert(0xFF & (material_code >> 16));
    97     s.surface_index = convert(0xFF & (material_code >> 8));
 


suggests correspondence:

* material1 is inner
* material2 is outer




