<?xml version='1.0' encoding='UTF-8'?>
<xsl:stylesheet version='1.0' xmlns:xsl='http://www.w3.org/1999/XSL/Transform' >
<xsl:output method='xml' version='1.0' encoding='UTF-8' indent='no'/>
<xsl:preserve-space elements="*" />

<xsl:param name="host">localhost</xsl:param>
<xsl:param name="livedir"></xsl:param>

<xsl:template match="/" >
  <xsl:apply-templates/>
</xsl:template>

<!-- 
 
    add the extension modules to the conf.xml 
    and change watchdoc params

<xsl:template match="xquery/builtin-modules/module[position()=last()]" >
<xsl:text>
</xsl:text>
<module uri="http://hfag.phys.ntu.edu.tw/hfagc/average" class="org.exist.xquery.modules.hfag.AverageModule"/>
<xsl:text>
</xsl:text>
<module uri="http://hfag.phys.ntu.edu.tw/hfagc/jima"    class="org.hfag.xquery.modules.jima.JimaAvgModule"/>
<xsl:text>
</xsl:text>
<module uri="http://hfag.phys.ntu.edu.tw/hfagc/perl"    class="org.hfag.xquery.modules.perl.PerlModule"/>
<xsl:text>
</xsl:text>
<module uri="http://hfag.phys.ntu.edu.tw/hfagc/ivue" class="org.hfag.xquery.modules.ivue.IvueModule"/>
<xsl:text>
</xsl:text>
<module uri="http://hfag.phys.ntu.edu.tw/hfagc/ixml" class="org.hfag.xquery.modules.ixml.IxmlModule"/>
<xsl:text>
</xsl:text>
</xsl:template>
--> 

<xsl:template match="watchdog/@output-size-limit" >
 <xsl:attribute name="output-size-limit" >30000</xsl:attribute>
</xsl:template>

<xsl:template match="serializer" >
  <xsl:comment>
      modify add-exist-id from none to element, for source doc in result fragments 
	  this results in /servlet/blah/smth.xml?_xpath=//*[local-name()='quote']
	  providing exist:id and exist:src attributes on the quote elements 
	  unfortunately the exist:src is smth.xml only rather than the complete
	  path 
	    trivial change to 
		   org.exist.storage.serializers.NatriveSerializer to provide the full
		 path , search for SOURCE_ATTRIB
  </xsl:comment><xsl:text>
  </xsl:text>
  <xsl:element name="serializer" >
     <xsl:attribute name="add-exist-id" >element</xsl:attribute> 
     <xsl:apply-templates select="@*[local-name()!='add-exist-id']" />
  </xsl:element>	 
</xsl:template>


<xsl:template match="cluster" >
  <xsl:comment>
     prefix journalDir with $livedir
   </xsl:comment><xsl:text>
  </xsl:text>
  <xsl:element name="cluster" >
     <xsl:attribute name="journalDir" ><xsl:value-of select="concat($livedir,'/',@journalDir)" /></xsl:attribute> 
     <xsl:apply-templates select="@*[local-name()!='journalDir']" />
  </xsl:element>	 
</xsl:template>

<xsl:template match="db-connection" >
  <xsl:comment>
     prefix files with $livedir
   </xsl:comment><xsl:text>
  </xsl:text>
  <xsl:element name="db-connection" >
     <xsl:attribute name="files" ><xsl:value-of select="concat($livedir,'/',@files)" /></xsl:attribute> 
     <xsl:apply-templates select="@*[local-name()!='files']" />
  </xsl:element>	 
</xsl:template>


<!-- This is a simple identity function -->
<xsl:template match="@*|*|text()|processing-instruction()|comment()"  >
  <xsl:copy>
    <xsl:apply-templates select="*|@*|text()|processing-instruction()|comment()" />
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>

