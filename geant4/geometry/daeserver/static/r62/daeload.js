/**

  * TODO:

    * bbox dimensions output
    * rotation controls 
    * canvas size
    * html links not working 
    * form controls, webpy POST
    * cull excessive siblings
    * transparency
    * html tree

   * http://belle7.nuu.edu.tw/dae/tree/3154.html

     * the AD is on its side : fixed by changing to Z_UP at DAECopy stage
   
   * http://belle7.nuu.edu.tw/dae/tree/3154.html?cam=0.1,0.1,0.1&anim=1&rotate=0,0.1,0

     * TODO: change to spec axis of rotation and delta ?
       
   * http://belle7.nuu.edu.tw/dae/tree/3154___0.html?cam=0.1,0.1,0.1

     * see nothing here, with camera at center : need to fix double sided 
     * now double sided and sides visble, but curious culling

  * http://belle7.nuu.edu.tw/dae/tree/3154.html?fov=0
  * http://belle7.nuu.edu.tw/dae/tree/2___1.html?cam=0.5,0.5,0.5&fov=0
  
    * frame filling from orthographic ?

   * http://belle7.nuu.edu.tw/dae/tree/3154___0.html?bbcam=1&cam=0,4,1&anim=1&fov=80&rotation=0.0,0.0,0.01
       
     * curious cut, hitting a edge ?

   * http://belle7.nuu.edu.tw/dae/tree/3153___0.html?bbcam=1&cam=0,4,1&anim=1&fov=80&rotation=0.0,0.0,0.01

     * evern wierder cutting 

   * http://localhost/dae/tree/3155___1.html?cam=2,2,2

     * depth is not enough to protect from heavy geometry for some volumes, 
       maybe could cull excessive siblings 

   * http://localhost/dae/tree/2___1.html?cam=0.5,0.5,0.5

   * http://belle7.nuu.edu.tw/dae/tree/0.html?cam=0.1,0.1,0.1&anim=1

     * with double sided on, the view from inside is kinda confusing



*/

THREE_enum_side = { 'double':THREE.DoubleSide, 'front':THREE.FrontSide, 'back':THREE.BackSide };
THREE_enum_bool = { '1':true , '0':false };
THREE_Vector3_fromString = function( s ){
   var xyz = s.split(",");
   x = parseFloat(xyz[0]) || 0 ;
   y = parseFloat(xyz[1]) || 0 ;
   z = parseFloat(xyz[2]) || 0 ;
   return new THREE.Vector3(x,y,z) ;
}


DAELOAD = function(){

        var param ;
        var renderer, width, height ; 
        var camera ;
        var geometry, material ;
        var mesh, scene ;
        var rotation ;
        var dae ; 

        function handle_load( collada ){
		    dae = collada.scene;
			init(param);

            var defaults = { anim:"0" , rotation:"0.01,0.01,0.01" };
            var anim = THREE_enum_bool[param.anim || defaults.anim] ;
            if( anim )
            {
                rotation = THREE_Vector3_fromString( param.rotation || defaults.rotation );
                animate();
            }
            else
            {
                render();
            }
        }

        function load( _param ){
            param = _param ;
            url = param.url || "../static/models/demo.dae"  ;
		    var loader = new THREE.ColladaLoader();
		    loader.options.convertUpAxis = true;
		    loader.load( url , handle_load );
        } 

        function init_renderer( param ) {

            var defaults = { id:"container" } ;
            var id = param.id || defaults.id ;

            var container = document.getElementById(id);
            width = container.offsetWidth ;
            height = container.offsetHeight ;

            renderer = new THREE.CanvasRenderer();
            renderer.setSize( width, height );
            container.appendChild( renderer.domElement );

        }


        function init_camera( param ){

            var defaults = { fov:"75", near:"0.1", far:"100", cam:"2,2,2", look:"0,0,0",  bbunit:"1" , orth:"10,10,1" };

            var fov = parseFloat(param.fov || defaults.fov) ;   // vertical fov in degrees
 
            var bbunit = THREE_enum_bool[param.bbunit || defaults.bbunit ];        // determines the unit of the positions

            var near = parseFloat(param.near || defaults.near) ;
            var far = parseFloat(param.far || defaults.far) ; 
            var cam = THREE_Vector3_fromString( param.cam || defaults.cam );
            var look = THREE_Vector3_fromString( param.look || defaults.look );
            var orth = THREE_Vector3_fromString( param.orth || defaults.orth );

            if ( bbunit ){
                //
                // when using `bbunit=1` the `cam=1,1,1` and look args are 
                // regarded to be in units 
                // of the bounding size obtained from the below for each dimension
                //
                // this allows the camera to positioned inside OR outside an
                // unknown geometry such that on rotating around the geometry should
                // never knock over the camera  
                //
                geometry.computeBoundingBox();
                var bb = geometry.boundingBox  ; 

                //  max extent along each axis
                //
                //  var bbsize = new THREE.Vector3( 
                //      Math.max(Math.abs(bb.max.x),Math.abs(bb.min.x)), 
                //      Math.max(Math.abs(bb.max.y),Math.abs(bb.min.y)), 
                //      Math.max(Math.abs(bb.max.z),Math.abs(bb.min.z))) ;
                //  cam.set( cam.x * bbsize.x , cam.y * bbsize.y , cam.z * bbsize.z );  

                // max extent along any axis
                var bbsize = Math.max(
                                Math.abs(bb.max.x),Math.abs(bb.min.x), 
                                Math.abs(bb.max.y),Math.abs(bb.min.y), 
                                Math.abs(bb.max.z),Math.abs(bb.min.z)
                             );

                cam.multiplyScalar( bbsize );
                look.multiplyScalar( bbsize );
                near = near * bbsize ;
                far = far * bbsize ;
            }


            if ( fov == 0 )
            {
                var right = orth.x * width * 0.5  ; 
                var top   = orth.y * height * 0.5 ; 
                camera = new THREE.OrthographicCamera( -right, right, top, -top, near, far );
            }
            else
            {
                camera = new THREE.PerspectiveCamera( fov, width / height, near, far );
            }


            camera.position.copy(cam);
            camera.lookAt(look);
        }

        function init_cube( param ){

            var defaults = { size:200, color:0xff0000, wireframe:"1", side:"double", obj:"0,0,0" };

            var obj = THREE_Vector3_fromString( param.obj || defaults.obj );
            var size = param.size || defaults.size ;
            var color = param.color || defaults.color
            var wireframe = THREE_enum_bool[param.wireframe || defaults.wireframe] ;
            var side = THREE_enum_side[param.side || defaults.side] ;  

            geometry = new THREE.CubeGeometry( size, size, size );
            material = new THREE.MeshBasicMaterial( { color: color, wireframe: wireframe } );
            material.side = side ; 
            mesh = new THREE.Mesh( geometry, material );  // fallback to cube
            mesh.position.copy(obj);
        }


        function init_loaded( param ){

            var defaults = { obj:"0,0,0" };

            var obj = THREE_Vector3_fromString( param.obj || defaults.obj );
            var root = dae ;
            var ppv = root.children[0] ; 
            var  lv = ppv.children[0] ; 

            mesh = lv ;           
            geometry = mesh.geometry ; 

            mesh.position.copy(obj);
        } 
     

        function instrument_object( obj , param ){

            var defaults = { axes:"1" , bbox:"1" };

            var axes = THREE_enum_bool[param.axes || defaults.axes] ;
            var bbox = THREE_enum_bool[param.bbox || defaults.bbox] ;

            if( axes ){

               if ( obj.geometry.boundingBox === null ) {
                    obj.geometry.computeBoundingBox();
               }
               var bb = obj.geometry.boundingBox ; 
               var size = Math.max( 
                          Math.abs(bb.min.x), Math.abs(bb.max.x), 
                          Math.abs(bb.min.y), Math.abs(bb.max.y), 
                          Math.abs(bb.min.z), Math.abs(bb.max.z)
                          ); 
                obj.add(new THREE.DoubleAxisHelper( size * 1.2 ));
                obj.add(new THREE.DoubleAxisHelper( size * -1.2 ));
            }

            if( bbox ){
                obj.add(new THREE.BoxHelper(obj)) ; 
            }
        }


        function init_scene( param ){
            scene = new THREE.Scene();
            instrument_object( mesh , param );
            scene.add( mesh ); 
        }


       /** r62 /usr/local/env/graphics/webgl/three.js/examples/webgl_loader_collada.html  
        */

        function init(param) {

            init_renderer( param ); 

            //init_cube( param );
            init_loaded(param) ;

            init_scene( param );
            init_camera( param );
       }

        function animate() {
            requestAnimationFrame( animate );
            mesh.rotation.x += rotation.x ;
            mesh.rotation.y += rotation.y ;
            mesh.rotation.z += rotation.z ;
            render();
        }

        function render(){
            renderer.render( scene, camera );
        }

        return {
            init : init,
            animate : animate, 
            load : load
        };


}();

