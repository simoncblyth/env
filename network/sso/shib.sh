#!/bin/sh

# https://gist.github.com/olberger/d34253822a84664648b3

#set -x

# Usage: shibb-cas-get.sh {username} {password} # If you have any errors try removing the redirects to get more information
# The service to be called, and a url-encoded version (the url encoding isn't perfect, if you're encoding complex stuff you may wish to replace with a different method)

DEST=https://myapp.example.com/
SP=https://myapp.example.com/index.php
IDP="https://myidp.example.com/idp/shibboleth&btn_sso=SSOok"

#Authentication details. This script only supports username/password login, but curl can handle certificate login if required
USERNAME=$1
PASSWORD=$2

#Temporary files used by curl to store cookies and http headers
COOKIE_JAR=.cookieJar
HEADER_DUMP_DEST=.headers
rm $COOKIE_JAR
rm $HEADER_DUMP_DEST

#The script itself is below

# 1. connect to the page which will display the WAYF

curl -s -k -c $COOKIE_JAR "$DEST" -o /dev/null

# 2. Choose the Shibb provider and get redirected to the CAS login form

#Visit CAS and get a login form. This includes a unique ID for the form, which we will store in CAS_ID and attach to our form submission. jsessionid cookie will be set here

CAS_ID=`curl -s -L --data "provider=$IDP" -i -b $COOKIE_JAR -c $COOKIE_JAR "$SP" -D $HEADER_DUMP_DEST  | grep name=.lt | sed 's/.*value..//' | sed 's/\".*//'`

dos2unix $HEADER_DUMP_DEST > /dev/null 2>&1

CURL_DEST=`grep Location $HEADER_DUMP_DEST | tail -n 1 | sed 's/Location: //'`

# 3. Submit the login form, using the cookies saved in the cookie jar and the form submission ID just extracted. We keep the headers from this request as the return value should be a 302 including a "ticket" param which we'll need in the next request

curl -L -s -k --data "username=$USERNAME&password=$PASSWORD&lt=$CAS_ID&_eventId=submit&submit=LOGIN" -i -b $COOKIE_JAR -c $COOKIE_JAR $CURL_DEST -D $HEADER_DUMP_DEST -o file.html

# 4. Extract the SAMLResponse and RelayState from the page and submit it
grep 'name=.SAMLResponse' file.html | sed 's/.*value..//' | sed 's/\".*//' > samlresponse.txt
relaystate=`grep 'name=.RelayState' file.html | sed 's/.*value..//' | sed 's/\".*//' | perl -MHTML::Entities -le 'while(<>) {print decode_entities($_);}'`
target=`grep 'form action.' file.html | sed 's/.*action..//' | sed 's/\".*//' | perl -MHTML::Entities -le 'while(<>) {print decode_entities($_);}'`

# Post the form and get redirected to the page
curl -L -s -k --data "RelayState=$relaystate" --data-urlencode SAMLResponse@samlresponse.txt -i -b $COOKIE_JAR -c $COOKIE_JAR "$target"
