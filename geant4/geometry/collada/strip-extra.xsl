<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output method="xml" indent="yes" encoding="ISO-8859-1"/>
 
<xsl:template match="*[local-name()='extra']" />
  
<xsl:template match="@*|node( )">
  <xsl:copy>
     <xsl:apply-templates select="@*|node( )"/>
   </xsl:copy>
</xsl:template>
	      
</xsl:stylesheet>
