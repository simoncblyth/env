Import-Module opticks -DisableNameChecking

$prefix = "$(opticks-prefix)\externals"

function dist-{     Import-Module dist -DisableNameChecking }
function dist-src{  "${env:userprofile}\env\psm1\dist\dist.psm1" }
function dist-vi{   vi $(dist-src) }
function dist-filename{  param([string]$url) ([System.Uri]$url).Segments[-1] }
function dist-ext{       param([string]$filename) [io.path]::GetExtension($filename) }
function dist-name{      param([string]$filename) [io.path]::GetFileNameWithoutExtension($filename) }
function dist-join{      param([string]$base,[string]$name)  [io.path]::combine($base,$name) }


function dist-info 
{
   param(
      [string]$tag,
      [string]$url, 
      [string]$override=""
   )

   $filename = dist-filename $url 
   $name = if($override -eq "") { dist-name $filename } else { $override }


   $ext = dist-ext $filename
   $fold = dist-join $prefix $tag
   $dir = dist-join $fold $name
   $zip = dist-join $fold $filename
   $abszip = "${env:homedrive}$zip"
   $absfold = "${env:homedrive}$fold"
   $bdir = $dir + ".build" 

   @{ 
      tag=$tag;
      url=$url;
      filename=$filename;
      ext=$ext;
      name=$name;
      override=$override;
      fold=$fold;
      dir=$dir;
      bdir=$bdir;
      zip=$zip;
      abszip=$abszip;
      absfold=$absfold;
      prefix=$prefix;
   }
}




function dist-get
{ 
   param(
      [string]$tag = "xercesc",
      [string]$url = "http://ftp.mirror.tw/pub/apache//xerces/c/3/sources/xerces-c-3.1.3.zip"
   )


   $info = dist-info $tag $url

   $filename = $info.Item("filename")
   $ext = $info.Item("ext")
   $name = $info.Item("name")
   $fold = $info.Item("fold")
   $zip = $info.Item("zip")
   $dir = $info.Item("dir")
   $abszip = $info.Item("abszip")
   $absfold = $info.Item("absfold")


   Write-Host @"
   
   tag : $tag
   url : $url
   prefix : $prefix

   filename :  $filename
   ext      :  $ext
   name     :  $name
   fold     :  $fold
   zip      :  $zip
   dir      :  $dir

   abszip   :  $abszip
   absfold  :  $absfold


"@

   mkdir -Force $fold > $null
   cd $fold

   if(![io.file]::exists($zip))
   {
       Write-Host "Downloading $url to $zip "
       $wc = New-Object System.Net.WebClient
       $wc.DownloadFile($url,$zip)
   }
   else
   {
       Write-Host "Already downloaded url to zip "

   }

   if(![io.directory]::exists($dir))
   {
       Write-Host "Extracting zip into dir "

       $shell = new-object -com shell.application
       $archive = $shell.namespace("$abszip")
       $dest = $shell.namespace("$absfold")
       $dest.CopyHere($archive.items())
   }
   else
   {
       Write-Host "Already extracted zip into dir "
   }
}


Export-ModuleMember -Function "dist-*"

