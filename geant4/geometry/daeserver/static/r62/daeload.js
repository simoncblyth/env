/**



   * http://belle7.nuu.edu.tw/dae/tree/3154___0.html?bbcam=1&cam=0,4,0&anim=0&fov=30&rotation=0.0,0.01,0.01

     * the AD is on its side : fix up axis
          
   * http://belle7.nuu.edu.tw/dae/tree/3154___0.html?bbcam=1&cam=0,0,0&anim=0&fov=30&rotation=0.0,0.01,0.01

     * see nothing here, with camera at center : need to fix double sided 

   * http://belle7.nuu.edu.tw/dae/tree/3154___0.html?bbcam=1&cam=0,4,1&anim=1&fov=80&rotation=0.0,0.0,0.01
       
     * curious cut, hitting a edge ?

   * http://belle7.nuu.edu.tw/dae/tree/3153___0.html?bbcam=1&cam=0,4,1&anim=1&fov=80&rotation=0.0,0.0,0.01

     * evern wierder cutting 
 
   * http://belle7.nuu.edu.tw/dae/tree/3153___0.html?bbcam=1&cam=0,4,1&anim=1&fov=50&rotation=0.0,0.0,0.01&far=1000

     * any setting of far makes it dissappear

   * http://belle7.nuu.edu.tw/dae/tree/3152___0.html?bbcam=1&cam=0,4,1&anim=1&fov=120&rotation=0.0,0.0,0.01
 
     * bizarre shark animation


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

            var defaults = { fov:75, near:1, far:10000, cam:"0,0,1000", look:"0,0,0",  bbcam:"0" };

            var fov = param.fov || defaults.fov ;
            var near = param.near || defaults.near ;
            var far = param.far || defaults.far ; 
            var cam = THREE_Vector3_fromString( param.cam || defaults.cam );
            var look = THREE_Vector3_fromString( param.look || defaults.look );

            var bbcam = THREE_enum_bool[param.bbcam || defaults.bbcam ];

            if ( bbcam ){
                //
                // when using `bbcam=1` the `cam=1,1,1` arg is regarded to be in units 
                // of the bounding size obtained from the below for each dimension
                //     `max(abs(bb.max),abs(bb.min))` 
                //
                // this allows the camera to positioned inside OR outside an
                // unknown geometry such that on rotating around the geometry should
                // never knock over the camera  
                //
                geometry.computeBoundingBox();
                var bb = geometry.boundingBox  ; 
                //var bbsize = new THREE.Vector3( bb.max.x - bb.min.x, bb.max.y - bb.min.y , bb.max.z - bb.min.z ); 
                var bbsize = new THREE.Vector3( 
                      Math.max(Math.abs(bb.max.x),Math.abs(bb.min.x)), 
                      Math.max(Math.abs(bb.max.y),Math.abs(bb.min.y)), 
                      Math.max(Math.abs(bb.max.z),Math.abs(bb.min.z))) ;

                cam.set( cam.x * bbsize.x , cam.y * bbsize.y , cam.z * bbsize.z );  
            }


            if ( fov == 0 )
            {
                var right = width / 2 ; 
                var left = - right ; 
                var top = height  ; 
                var bottom = - top ; 
                camera = new THREE.OrthographicCamera( left, right, top, bottom, near, far );
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

     
        function init_scene( param ){
            scene = new THREE.Scene();
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

