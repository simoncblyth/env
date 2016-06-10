function HelloWorld
{
     Write-Host "Hello World"


     choco list --local-only

     cmake --version

     git --version

     hg --version
}



function GoTo-Website
{
    Param ($website)
    $global:ie.navigate($website);
    $global:image = @($ie.Document.getElementByName("main_image"))[0].href;
    $global:title = @($ie.Document.getElementByTagName("h1"))[3].innerHTML;
    $global:date = @($ie.Document.getElementByTagName("h3"))[0].innerHTML;
}



