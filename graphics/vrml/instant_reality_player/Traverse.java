/*

http://dcgi.felk.cvut.cz/home/zara/courses/EAI2/CreateTest/CreateTest.java

http://tecfa.unige.ch/guides/vrml/vrmlman/node28.html

*/
class Traverse
{
	private static float[] red   = { 1, 0, 0 };
	public static void main(String[] args)
	{
		vrml.eai.Browser browser = null;
		try
		{
			java.net.InetAddress address = java.net.InetAddress.getByName("localhost");
			browser = vrml.eai.BrowserFactory.getBrowser(address, 4848);

			System.out.println("Browser.Name = \"" + browser.getName() + '"');
			System.out.println("Browser.Version = \"" + browser.getVersion() + '"');
			System.out.println("Browser.CurrentSpeed = " + browser.getCurrentSpeed());
			System.out.println("Browser.CurrentFrameRate = " + browser.getCurrentFrameRate());
			System.out.println("Browser.WorldURL = \"" + browser.getWorldURL() + '"');
  
            browser.beginUpdate();
            vrml.eai.Node root = browser.getNode("root"); // suspect implicit "scene" root node
            traverse((vrml.eai.field.EventOutMFNode) root.getEventOut("children"));
            browser.endUpdate();
 
		}
		catch (Throwable all)
		{
			all.printStackTrace();
		}
		finally
		{
			if (browser != null)
				browser.dispose();
		}
	}

    public static void traverse(vrml.eai.field.EventOutMFNode children){

        if (children == null) return;
        vrml.eai.Node nodes[]  = children.getValue();
        int num_nodes = nodes.length;
 
        for (int i = 0; i < num_nodes; ++i) {
            String node_type = nodes[i].getType();
            System.out.println(i + " " + node_type );
            if (node_type.compareTo("Shape") == 0){
                // no way to navigate from a shape to its appearance ? there is a way just its painful

                //vrml.eai.field.EventInSFNode appIn = (vrml.eai.field.EventInSFNode) nodes[i].getEventIn("set_appearance");
		        //System.out.println("appIn = \"" + appIn + '"');

                vrml.eai.field.EventOutSFNode appOut = (vrml.eai.field.EventOutSFNode) nodes[i].getEventOut("appearance");
                vrml.eai.Node app = appOut.getValue() ;
		        System.out.println("appOut = \"" + appOut + '"');
		        System.out.println("app = \"" + app + '"');

                // events provide the way in/out of the scene, have to get or set their values to get/set anything

                vrml.eai.field.EventOutSFNode matOut = (vrml.eai.field.EventOutSFNode) app.getEventOut("material");
                vrml.eai.Node mat = matOut.getValue() ;
                vrml.eai.field.EventOutSFColor diffuseColorOut = (vrml.eai.field.EventOutSFColor)mat.getEventOut("diffuseColor_changed");
                
                float diffuseColor[] = diffuseColorOut.getValue();

		        System.out.println("matOut = \"" + matOut + '"');
		        System.out.println("mat = \"" + mat + '"');
		        System.out.println("diffuseColor = \"" + diffuseColor[0] +","+diffuseColor[1]+","+diffuseColor[2] + '"');

                vrml.eai.field.EventInSFColor diffuseColorIn = (vrml.eai.field.EventInSFColor)mat.getEventIn("set_diffuseColor");
                diffuseColorIn.setValue(red); 

                vrml.eai.field.EventInSFColor emissiveColorIn = (vrml.eai.field.EventInSFColor)mat.getEventIn("set_emissiveColor");
                emissiveColorIn.setValue(red); 

            } 

        }
    }
}
