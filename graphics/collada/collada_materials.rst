Collada Materials
====================

Geometry is associated with material within ``instance_geometry``, 
symbol/target pairs declare key/values of the material mapping::


    427   <library_nodes>
    428     <node id="__dd__Geometry__Sites__lvNearHallTop0xbf55760" name="__dd__Geometry__Sites__lvNearHallTop0xbf55760">
    429       <instance_geometry url="#near_hall_top_dwarf0xbbd1a68">
    430         <bind_material>
    431           <technique_common>
    432             <instance_material symbol="WHITE" target="#__dd__Materials__Air0xbbd2908" />
    433           </technique_common>
    434         </bind_material>
    435       </instance_geometry>
    436     </node>

The symbol keys are subsequntly used by the polylist ``material`` attribute.::

    168         <vertices id="WorldBox0xbbf06d8-Vtx">
    169           <input semantic="POSITION" source="#WorldBox0xbbf06d8-Pos" />
    170         </vertices>
    171         <polylist count="6" material="WHITE">
    172           <input offset="0" semantic="VERTEX" source="#WorldBox0xbbf06d8-Vtx" />
    173           <input offset="1" semantic="NORMAL" source="#WorldBox0xbbf06d8-Norm" />
    174           <vcount>4 4 4 4 4 4 </vcount>
    175           <p>0 0  3 0  2 0  1 0   4 1  7 1  3 1  0 1   7 2  6 2  2 2  3 2   6 3  5 3  1 3  2 3   5 4  4 4  0 4  1 4   4 5  5 5  6 5  7 5   </p>
    176         </polylist>


Basis effects use phong with simple color attributes.


MeshLab material loading
--------------------------

Code looks to deal with image textures only not phong shaders.

Threejs Loading
----------------

Some handling is present for phong etc..







