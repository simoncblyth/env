Javascript Lib Structure
==========================

* http://stackoverflow.com/questions/2190801/passing-parameters-to-javascript-files

file.js::

    var MYLIBRARY = MYLIBRARY || (function(){
        var _args = {}; // private

        return {
            init : function(Args) {
                _args = Args;
                // some other initialising
            },
            helloWorld : function() {
                alert('Hello World! -' + _args[0]);
            }
        };
    }());

html::

    <script type="text/javascript" src="file.js"></script>
    <script type="text/javascript">
       MYLIBRARY.init(["somevalue", 1, "controlId"]);
       MYLIBRARY.helloWorld();
    </script>


* http://geekswithblogs.net/PhubarBaz/archive/2011/11/21/getting-query-parameters-in-javascript.aspx

::

    // Creates associative array (object) of query params
    var QueryParameters = (function()
    {
        var result = {};

        if (window.location.search)
        {
            // split up the query string and store in an associative array
            var params = window.location.search.slice(1).split("&");
            for (var i = 0; i < params.length; i++)
            {
                var tmp = params[i].split("=");
                result[tmp[0]] = unescape(tmp[1]);
            }
        }

        return result;
    }());


    var debug = (QueryParameters.debug === "true");




