class EAIExample
{
	private static void onBrowserChanged(vrml.eai.event.BrowserEvent evt)
	{
		// Exit the program when there is an error or InstantPlayer gets closed
		switch (evt.getID())
		{
		case vrml.eai.event.BrowserEvent.INITIALIZED:
			break;
		case vrml.eai.event.BrowserEvent.SHUTDOWN:
		case vrml.eai.event.BrowserEvent.URL_ERROR:
		case vrml.eai.event.BrowserEvent.CONNECTION_ERROR:
		default:
			System.exit(0);
		}
	}

	private static float[] red   = { 1, 0, 0 };
	private static float[] green = { 0, 1, 0 };
	private static vrml.eai.field.EventInSFColor set_diffuseColor;

	private static void onIsOverChanged(vrml.eai.event.VrmlEvent evt)
	{
		// Change the color of the sphere to red when the mouse pointer is over the
		// sphere, and back to green when it is not
		vrml.eai.field.EventOutSFBool isOver = (vrml.eai.field.EventOutSFBool)evt.getSource();
		set_diffuseColor.setValue(isOver.getValue() == true ? red : green);
	}

	public static void main(String[] args)
	{
		vrml.eai.Browser browser = null;
		try
		{
			// Initialize the connection
			java.net.InetAddress address = java.net.InetAddress.getByName("localhost");
			browser = vrml.eai.BrowserFactory.getBrowser(address, 4848);

			// Add a listener to the browser. The listener is an instance of an anonymous class that
			// inherits from vrml.eai.event.BrowserListener and simply calls our onBrowserChanged method
			browser.addBrowserListener(
					new vrml.eai.event.BrowserListener()
					{
						public void browserChanged(vrml.eai.event.BrowserEvent evt)
						{
							EAIExample.onBrowserChanged(evt);
						}
					}
				);

			// Get the isOver event out of the TouchSensor node
			vrml.eai.Node touchSensor = browser.getNode("touchSensor");
			vrml.eai.field.EventOutSFBool isOver = (vrml.eai.field.EventOutSFBool)touchSensor.getEventOut("isOver");

			// Get the set_diffuseColor event in of the Material node
			vrml.eai.Node material = browser.getNode("material");
			set_diffuseColor = (vrml.eai.field.EventInSFColor)material.getEventIn("set_diffuseColor");

			// Add a listener to the isOver event out. The listener is an instance of an anonymous class
			// that inherits from vrml.eai.event.VrmlEventListener and simply calls our onIsOverChanged method
			isOver.addVrmlEventListener(
					new vrml.eai.event.VrmlEventListener()
					{
						public void eventOutChanged(vrml.eai.event.VrmlEvent evt)
						{
							EAIExample.onIsOverChanged(evt);
						}
					}
				);

			// Wait forever
			while (true)
				Thread.sleep(1000);
		}
		catch (Throwable all)
		{
			all.printStackTrace();
		}
		finally
		{
			// Shutdown
			if (browser != null)
				browser.dispose();
		}
	}
}
