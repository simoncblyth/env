<?xml version='1.0' encoding='UTF-8'?>
<xsl:stylesheet version='1.0' xmlns:xsl='http://www.w3.org/1999/XSL/Transform' >
<xsl:output method='xml' version='1.0' encoding='UTF-8' indent='no' 
   doctype-public="-//Mort Bay Consulting//DTD Configure 1.2//EN"  
   doctype-system="http://jetty.mortbay.org/configure_1_2.dtd"
   />

<xsl:preserve-space elements="*" />
<xsl:param name="jkport">8009</xsl:param>
<xsl:param name="port">8080</xsl:param>
<xsl:param name="context">exist</xsl:param>

<xsl:template match="/" >
  <xsl:apply-templates/>
</xsl:template>

<xsl:template match="SystemProperty[@name='jetty.port']" >
   <xsl:comment>Customized via stylesheet src/jetty.xsl </xsl:comment>
   <xsl:element name="SystemProperty" >
      <xsl:attribute name="name"  >jetty.port</xsl:attribute>
      <xsl:attribute name="default"><xsl:value-of select="$port"/></xsl:attribute>
   </xsl:element>
</xsl:template>

<xsl:template match="Call[@name='addWebApplication']/Arg[1]" >
    <xsl:comment>Customized via stylesheet src/jetty.xsl</xsl:comment>
	<xsl:element name="Arg">
	   <xsl:value-of select="concat('/',$context)"/>
	</xsl:element>
</xsl:template>

<xsl:template match="Call[@name='addWebApplication']" >

    <xsl:comment> add the ajp listener for mod_jk functionality, via src/jetty.xsl  </xsl:comment>

    <xsl:element name="Call" >
	   <xsl:attribute name="name" >addListener</xsl:attribute>
       <xsl:element name="Arg" >
	      <New class="org.mortbay.http.ajp.AJP13Listener">
		    <Set name="Port"><xsl:value-of select="$jkport"/></Set>
			<Set name="MinThreads">5</Set>
			<Set name="MaxThreads">20</Set>
			<Set name="MaxIdleTimeMs">0</Set>
			<Set name="confidentialPort">443</Set>
		 </New>
		</xsl:element>
	</xsl:element>


    <xsl:element name="Call" >
	   <xsl:attribute name="name" >addWebApplication</xsl:attribute>
	   <xsl:apply-templates/>
	</xsl:element>

</xsl:template>

<xsl:template match="Call[@name='addWebApplication']/Arg[2]" >
    <xsl:comment>Customized via stylesheet src/jetty.xsl</xsl:comment>
	<xsl:element name="Arg">
	   <SystemProperty name="webapp.home" default="../.." />/webapp
	</xsl:element>
</xsl:template>

<xsl:template match="Set[@name='displayName']" >
    <xsl:comment>Customized via stylesheet src/jetty.xsl</xsl:comment>
	<xsl:element name="Set">
        <xsl:attribute name="name"  >displayName</xsl:attribute>
	    <xsl:value-of select="$context" />
	</xsl:element>
</xsl:template>



<!-- This is a simple identity function -->
<xsl:template match="@*|*|text()|processing-instruction()|comment()"  >
  <xsl:copy>
    <xsl:apply-templates select="*|@*|text()|processing-instruction()|comment()" />
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>

