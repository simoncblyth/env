# === func-gen- : comms/slack/slack fgp comms/slack/slack.bash fgn slack fgh comms/slack
slack-src(){      echo comms/slack/slack.bash ; }
slack-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slack-src)} ; }
slack-vi(){       vi $(slack-source) ; }
slack-env(){      elocal- ; }
slack-usage(){ cat << EOU

Slack : team messaging
=========================


Review
---------

* https://www.pcmag.com/article2/0,2817,2477507,00.asp

* https://reviews.financesonline.com/p/slack/

  Many integrations, eg with zendesk, mailchimp



* https://alternativeto.net/software/google-groups/



Stride
--------

* https://www.stride.com


Google Groups Alt
---------------------

* https://www.quora.com/Are-there-any-good-alternatives-to-Google-Groups


Mobilize
   free tier has no mailing list 

Slack
   https://slack.com 


Slack vs Google Groups (GG)
----------------------------

* https://www.quora.com/Is-Google-groups-better-than-slack

GG is publically accessible (google indexed) forum, 
Slack tends to be private. 


* https://lifehacker.com/drop-google-and-facebook-groups-and-use-this-instead-1823994067


* https://hiverhq.com/blog/alternative-google-groups/

  Inside Gmail web : so no go from China


* https://news.ycombinator.com/item?id=9410250

* https://www.discourse.org/


Groups.io
------------

* https://groups.io/

* https://groups.io/static/help

* https://groups.io/g/xephem

* https://groups.io/search?q=open+source

* https://wingedpig.com/2014/09/23/introducing-groups-io/
* https://wingedpig.com/2014/11/06/what-runs-groups-io/



Impressions on Slack and Alt (from reviews)
---------------------------------------------


Slack
   encourages lots of short messages

Google Groups
    traditional usenet, i like the public visibility 
    and meaninful urls : https://groups.google.com/forum/#!forum/openvdb-forum 

Flock

Atlassian JIRA : Stride

Twist 
    aims at distributed teams, closer to email

    https://www.pcmag.com/review/354433/twist

    https://twistapp.com

    https://twist.zendesk.com/hc/en-us

    https://www.techrepublic.com/article/how-twist-aims-to-compete-against-slack-with-its-own-approach-to-collaboration/


EOU
}
slack-dir(){ echo $(local-base)/env/comms/slack/comms/slack-slack ; }
slack-cd(){  cd $(slack-dir); }
slack-mate(){ mate $(slack-dir) ; }
slack-get(){
   local dir=$(dirname $(slack-dir)) &&  mkdir -p $dir && cd $dir

}
