class EAIFramework
{

	private static float[] red   = { 1, 0, 0 };
	private static float[] green = { 0, 1, 0 };
	private static vrml.eai.field.EventInSFColor set_diffuseColor;
	private static vrml.eai.field.EventOutSFBool showBBox_changed ;
	private static vrml.eai.field.EventInSFBool  set_showBBox ;

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

            vrml.eai.Node node = browser.getNode("S4425") ;
			System.out.println("Browser.WorldURL = \"" + node + '"');

            showBBox_changed = (vrml.eai.field.EventOutSFBool)node.getEventOut("showBBox_changed");

            Boolean showBBox_changed_v = showBBox_changed.getValue() ;
		    System.out.println("showBBox_changed = \"" + showBBox_changed + '"' + showBBox_changed_v );

            set_showBBox = (vrml.eai.field.EventInSFBool)node.getEventIn("set_showBBox");
		    System.out.println("set_showBBox = \"" + set_showBBox + '"');
            set_showBBox.setValue(!showBBox_changed_v );   

            /*
                flip flop the BBox of the named node being shown
                can also do from commandline:

                   curl "http://localhost:35668/setFieldValue?node=S4425&field=5&value=TRUE&link=referer"

                Avalon is handy for introspection

                   http://localhost:35668/Node.html?node=S4425
            */ 

           
             vrml.eai.Node appearance = node.getNode("appearance");
		     System.out.println("appearance = \"" + appearance + '"');


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
}
