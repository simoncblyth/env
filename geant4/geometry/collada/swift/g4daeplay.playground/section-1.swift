/*  See g4daeplay-  */

import Cocoa
import SceneKit
import XCPlayground
import QuartzCore

func addSpin(node: SCNNode)
{
    var spin = CABasicAnimation(keyPath: "rotation")
    spin.toValue = NSValue(SCNVector4: SCNVector4(x: 1, y: 1, z: 0, w: 2.0*M_PI))
    spin.duration = 3
    spin.repeatCount = HUGE
    node.addAnimation(spin, forKey: "spin around")
}

func make_torusNode() -> SCNNode
{
   var torus = SCNTorus(ringRadius: 1, pipeRadius: 0.35)
   var torusNode = SCNNode(geometry: torus)
   torus.firstMaterial.diffuse.contents  = NSColor.redColor()
   torus.firstMaterial.specular.contents = NSColor.whiteColor()
   return torusNode
}

func v_fmt(label:String, vec:SCNVector3)
{
    let nf = NSNumberFormatter()
    nf.numberStyle = NSNumberFormatterStyle.DecimalStyle
    nf.maximumFractionDigits = 2
    
    let x = nf.stringFromNumber(vec.x)
    let y = nf.stringFromNumber(vec.y)
    let z = nf.stringFromNumber(vec.z)
    
    println(" \(label) \(x)  \(y) \(z) ")
}

func v_sub(a:SCNVector3, b:SCNVector3) -> SCNVector3
{
    return SCNVector3Make(a.x-b.x, a.y-b.y,a.z-b.z)
}
func v_add(a:SCNVector3, b:SCNVector3) -> SCNVector3
{
    return SCNVector3Make(a.x+b.x, a.y+b.y,a.z+b.z)
}
func v_mul(a:SCNVector3, s:Double) -> SCNVector3
{
    return SCNVector3Make(a.x*s, a.y*s,a.z*s)
}
func v_avg(a:SCNVector3, b:SCNVector3)-> SCNVector3
{
    return v_mul(v_add(a,b),0.5)
}

func v_ext( node:SCNNode) -> SCNVector3
{
    var min = SCNVector3Make(0,0,0)
    var max = SCNVector3Make(0,0,0)
    node.getBoundingBoxMin(&min, max: &max)
    var ext = v_sub(max,min)
    
    v_fmt("min",min)
    v_fmt("max",max)
    v_fmt("ext",ext)
    
    return ext
}


func v_bbox( vol:SCNBoundingVolume )
{
    var min = SCNVector3Make(0,0,0)
    var max = SCNVector3Make(0,0,0)
    vol.getBoundingBoxMin(&min, max: &max)
    var ext = v_sub(max,min)
    
    v_fmt("min",min)
    v_fmt("max",max)
    v_fmt("ext",ext)
}


func v_relative_position( node:SCNNode, place:SCNVector3) -> SCNVector3
{
    var pos = node.position
    var ext = v_ext(node)
    var rel = SCNVector3Make( pos.x + place.x*ext.x,
                           pos.y + place.y*ext.y,
                           pos.z + place.z*ext.z )
    
    v_fmt("pos",pos)
    v_fmt("rel",rel)
    return rel
}





class DAE {
    
    var sceneView : SCNView
    var scene : SCNScene
    var sceneSource : SCNSceneSource?
    var cameraNode : SCNNode?
    var camera: SCNCamera?
    var nid : Array<String>?
    
    init(path: String, target: Int)
    {
        let url = NSURL(fileURLWithPath:path)
        sceneSource = SCNSceneSource(URL:url, options:nil)
        nid = sceneSource!.identifiersOfEntriesWithClass(SCNNode.self) as? [String]
        
        let nidCount = self.nid?.count
        println("sceneSource path \(path) nidCount \(nidCount) " )
        
        scene = SCNScene()
        
        sceneView = SCNView(frame: CGRect(x: 0, y: 0, width: 640, height: 640))
        sceneView.scene = scene
        //sceneView.backgroundColor = NSColor.grayColor()
        sceneView.autoenablesDefaultLighting = true
        //sceneView.allowsCameraControl = false
        
        let targetNode = self.get_node(target)
        
        self.add_node(targetNode!)
        self.lookAt(targetNode!)
   }

    
    
    func lookAt(targetNode: SCNNode)
    {
        
        camera = SCNCamera()
        camera!.orthographicScale = 3000
        camera!.usesOrthographicProjection = false
        camera!.zNear = 10
        camera!.zFar = 100000
        camera!.yFov = 80
        
        
        cameraNode = SCNNode()
        cameraNode!.camera = camera
        cameraNode!.position = SCNVector3Make(100,1000,1000.0)
        
        scene.rootNode.addChildNode(cameraNode)

        
        let lookAt = SCNLookAtConstraint(target:targetNode)
        
        cameraNode!.constraints = [lookAt]
        
    }
    
    
    
    
    func get_nid(index: Int)->String?
    {
        if let nodeIDs = nid {
            return nodeIDs[index]
        } else {
            return nil
        }
    }

    
    func get_geometry(name:String) -> SCNGeometry?
    {
        if let geom = sceneSource!.entryWithIdentifier(name, withClass:SCNGeometry.self) as? SCNGeometry
        {
            //geom.firstMaterial.diffuse.contents = NSColor.redColor()
            
            return geom
        } else {
            return nil
        }
    }
    
    
    func get_node(name: String) -> SCNNode?
    {
        if let theNode = sceneSource!.entryWithIdentifier(name, withClass:SCNNode.self) as? SCNNode
        {
            //println(theNode.position.x)
            return theNode
        } else {
            return nil
        }
    }
    func get_node( index: Int) -> SCNNode?
    {

        if index == -1 {
            return make_torusNode()
        }
        
        if let theNodeName = self.get_nid(index){
        println("index \(index) theNodeName \(theNodeName)")
        if let theNode = self.get_node(theNodeName){
            return theNode
        }
        }
        return nil
    }
    func childNode( name:String, recursively:Bool) -> SCNNode?
    {
        return self.scene.rootNode.childNodeWithName(name, recursively: recursively)
    }
    func sourceChildNode( name:String )->SCNNode?
    {
        return sceneSource!.entryWithIdentifier(name,withClass: SCNNode.self) as? SCNNode
    }
    func add_node( node:SCNNode)
    {
        scene.rootNode.addChildNode(node)
    }
    
}





let env = NSProcessInfo.processInfo().environment
let path: String? = env["DAE_NAME_AD"] as? NSString
if path {
    let dae = DAE(path:path!,target:0)
    XCPShowView("The Scene View", dae.sceneView)

    let pov = dae.sceneView.pointOfView
    pov.position
}

















