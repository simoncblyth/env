NAN Abort Photon
=================

* Resolved by special casing perfect normal incidence



Initially was numbering as index 279, but that is with time ordering sort on load, 
which with all mock photons starting at 1ns means randomly ordered.
Actially is mock photon index 513 when lpos is (0,0,1500) ie shoot from 1.5m infront of PMT::

     index 511 pos (-19104.3,-796636,-9297.4) dir (-0,0,-1) pol (0,0,1) _t 1 _wavelength 550 _pmtid 0x1051910
     index 512 pos (-17737.6,-801707,-9297.4) dir (-0,0,-1) pol (0,0,1) _t 1 _wavelength 550 _pmtid 0x1051911

     index 513 pos (-16650.4,-803385,-9297.4) dir (-0,0,-1) pol (0,0,1) _t 1 _wavelength 550 _pmtid 0x1051912

     index 514 pos (-15302.4,-802513,-9297.4) dir (-0,0,-1) pol (0,0,1) _t 1 _wavelength 550 _pmtid 0x1051913
     index 515 pos (-16389.6,-800835,-9297.4) dir (-0,0,-1) pol (0,0,1) _t 1 _wavelength 550 _pmtid 0x1051914
     index 516 pos (-16520,-802110,-9297.4) dir (-0,0,-1) pol (0,0,1) _t 1 _wavelength 550 _pmtid 0x1051915

The PMT 0x1051912 is in water shield on mid-line between ADs and underneath. The 
Looks like photon getting stuck on AD table ?

::

    delta:~ blyth$ udp.py --target 0x1051912
    sending [--target 0x1051912] to host:port delta.local:15006 



::

    In [8]: np.where(a.history & 1<<31)
    Out[8]: (array([513]),)

    In [9]: a.dump(513)
    [[ -16650.424 -803385.25    -9610.          2.404]
     [        nan         nan         nan     550.   ]
     [      0.          0.          1.          1.   ]
     [      0.          0.         -0.          0.   ]]
    photonid:  513
    history:  2147483713
    pmtid:  0


::

    g4daeview.sh --debugkernel --debugphoton 279  ## wrong one
    udp.py --debugphoton 513  # nope not handled


    g4daeview.sh --debugkernel --debugphoton 513   ## restart with correct one

    mocknuwa-- MOCK                                ## sling the MOCK photons at it 


::

    2014-11-26 17:13:11,863 INFO    env.geant4.geometry.collada.g4daeview.daephotonspropagator:149 nwork 684 step 0 max_steps 30 nsteps 30 
    2014-11-26 17:13:11,863 INFO    env.geant4.geometry.collada.g4daeview.daephotonspropagator:157 prepared_call first_photon 0 photons_this_round 684 nsteps 30 
    2014-11-26 17:13:11,863 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:262 _get_rng_states
    2014-11-26 17:13:11,863 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:226 setup_rng_states using seed 0 
    [  1]    513 material_code 421004288 inner 25 outer 24 si 4 ri1 1.346476 ri2 1.000000 abs 15077.077148 sca 1000000.000000 rem 0.000000 ncdf 38 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    FILL_STATE       START    [   513] slot  1 steps  1 lht 1799018 tpos    1.000  -16650.42 -803385.25   -9297.40    w  550.00   dir    -0.00     0.00    -1.00 pol    0.000    0.000    1.000 
    TO_BOUNDARY      PASS     [   513] slot -1 steps  1 lht 1799018 tpos    2.404  -16650.42 -803385.25   -9610.00    w  550.00   dir    -0.00     0.00    -1.00 pol    0.000    0.000    1.000 
    AT_SURFACE       CONTINUE [   513] slot -1 steps  1 lht 1799018 tpos    2.404  -16650.42 -803385.25   -9610.00    w  550.00   dir      nan      nan      nan pol    0.000    0.000    1.000 REFLECT_SPECULAR 
    2014-11-26 17:13:12,577 INFO    env.geant4.geometry.collada.g4daeview.daephotonspropagator:185 launch sequence times [0.28745388793945315] 



::

    630        if (s.surface_index != -1)
    631        {
    632            // **propagate_at_surface**  for default surface model
    633            //
    634            // #. SURFACE_ABSORB(BREAK)
    635            // #. SURFACE_DETECT(BREAK)
    636            // #. REFLECT_DIFFUSE(CONTINUE) .direction .polarization
    637            // #. REFLECT_SPECULAR(CONTINUE) .direction
    638            // #. NO other option, so never PASS? 
    639            //
    640            command = propagate_at_surface(p, s, rng, g, use_weights);
    641            status = STATUS_AT_SURFACE ;
    642 
    643            PDUMP( p, photon_id, status, steps, command, -1);
    644 
    645            if (command == BREAK) break ;
    646            if (command == CONTINUE) continue;
    647 
    648            status = STATUS_AT_SURFACE_UNEXPECTED ;
    649            PDUMP_ALL( p, photon_id, status, steps, command, -1);
    650        }



Problem is the AD table is horizontal and photon direction vertically down
so arrives at normal incidence with incident_angle zero.

::

    464 __device__ int
    465 propagate_at_specular_reflector(Photon &p, State &s)
    466 {
    467     float incident_angle = get_theta(s.surface_normal, -p.direction);
    468     float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    469     incident_plane_normal /= norm(incident_plane_normal);
    470 
    471     p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
    472 
    473     p.history |= REFLECT_SPECULAR;
    474     
    475     
    476 #ifdef VBO_DEBUG
    477     if( p.id == 513 )
    478     {
    479         printf("id %d incident_angle %f    %f %f %f \n", p.id, incident_angle, s.surface_normal.x, s.surface_normal.y, s.surface_normal.z );
    480     }
    481 #endif
    482     
    483     
    484     return CONTINUE;
    485 } // propagate_at_specular_reflector


::

    014-11-26 17:37:48,041 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:262 _get_rng_states
    2014-11-26 17:37:48,042 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:226 setup_rng_states using seed 0 
    [  1]    513 material_code 421004288 inner 25 outer 24 si 4 ri1 1.346476 ri2 1.000000 abs 15077.077148 sca 1000000.000000 rem 0.000000 ncdf 38 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    FILL_STATE       START    [   513] slot  1 steps  1 lht 1799018 tpos    1.000  -16650.42 -803385.25   -9297.40    w  550.00   dir    -0.00     0.00    -1.00 pol    0.000    0.000    1.000 
    TO_BOUNDARY      PASS     [   513] slot -1 steps  1 lht 1799018 tpos    2.404  -16650.42 -803385.25   -9610.00    w  550.00   dir    -0.00     0.00    -1.00 pol    0.000    0.000    1.000 
    id 513 incident_angle 0.000000    0.000000 -0.000000 1.000000 
    AT_SURFACE       CONTINUE [   513] slot -1 steps  1 lht 1799018 tpos    2.404  -16650.42 -803385.25   -9610.00    w  550.00   dir      nan      nan      nan pol    0.000    0.000    1.000 REFLECT_SPECULAR 
    2014-11-26 17:37:48,981 INFO    env.geant4.geometry.collada.g4daeview.daephotonspropagator:185 launch sequence times [0.46804925537109376] 



::

    2014-11-26 18:03:03,368 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:262 _get_rng_states
    2014-11-26 18:03:03,368 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:226 setup_rng_states using seed 0 
    [  1]    513 material_code 421004288 inner 25 outer 24 si 4 ri1 1.346476 ri2 1.000000 abs 15077.077148 sca 1000000.000000 rem 0.000000 ncdf 38 w0 60.000000 st 20.000000 cdf lo/up 0.000000 0.000000 
    FILL_STATE       START    [   513] slot  1 steps  1 lht 1799018 tpos    1.000  -16650.42 -803385.25   -9297.40    w  550.00   dir    -0.00     0.00    -1.00 pol    0.000    0.000    1.000 
    TO_BOUNDARY      PASS     [   513] slot -1 steps  1 lht 1799018 tpos    2.404  -16650.42 -803385.25   -9610.00    w  550.00   dir    -0.00     0.00    -1.00 pol    0.000    0.000    1.000 
    id 513 incident_angle          0.000000 
    id 513 s.surface_normal        0.000000 -0.000000 1.000000 
    id 513 p.direction             -0.000000 0.000000 -1.000000 
    id 513 incident_plane_normal   0.000000 0.000000 0.000000 
    id 513 norm(incident_plane_..) 0.000000 
    id 513 incident_plane_normal   nan nan nan 
    id 513 p.direction             nan nan nan 
    AT_SURFACE       CONTINUE [   513] slot -1 steps  1 lht 1799018 tpos    2.404  -16650.42 -803385.25   -9610.00    w  550.00   dir      nan      nan      nan pol    0.000    0.000    1.000 REFLECT_SPECULAR 
    2014-11-26 18:03:04,376 INFO    env.geant4.geometry.collada.g4daeview.daephotonspropagator:185 launch sequence times [0.5438138427734375] 
    2014-11-26 18:03:04,382 WARNING env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:450 cannot write_propagated with event that has not been saved to file and subsequently loaded
    2014-11-26 18:03:04,383 INFO    env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:201 _last_index



