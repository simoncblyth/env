# === func-gen- : tools/llm fgp tools/llm.bash fgn llm fgh tools src base/func.bash
llm-source(){   echo ${BASH_SOURCE} ; }
llm-edir(){ echo $(dirname $(llm-source)) ; }
llm-ecd(){  cd $(llm-edir); }
llm-dir(){  echo $LOCAL_BASE/env/tools/llm ; }
llm-cd(){   cd $(llm-dir); }
llm-vi(){   vi $(llm-source) ; }
llm-env(){  elocal- ; }
llm-usage(){ cat << EOU


https://simonwillison.net/2024/Jun/17/cli-language-models/


* https://llm.datasette.io/en/stable/

* https://simonwillison.net/2024/Jun/17/cli-language-models/


* https://github.com/simonw/llm

* https://github.com/simonw/llm-mlx

* https://simonwillison.net/tags/llm/

* https://simonwillison.net/2025/May/2/qwen3-8b/

* https://medium.com/@billcava/terminal-ai-how-llm-changed-my-workflow-71ef97ddab5b



Zeta Install llm with uv
-------------------------

* https://llm.datasette.io/en/stable/setup.html

::

    zeta:~ blyth$ uv tool install llm
    Resolved 32 packages in 4.06s
    Prepared 32 packages in 2.30s
    Installed 32 packages in 34ms
     + annotated-types==0.7.0
     + anyio==4.10.0
     + certifi==2025.8.3
     + click==8.2.1
     + click-default-group==1.2.4
     + condense-json==0.1.3
     + distro==1.9.0
     + h11==0.16.0
     + httpcore==1.0.9
     + httpx==0.28.1
     + idna==3.10
     + jiter==0.10.0
     + llm==0.27.1
     + openai==1.104.2
     + pip==25.2
     + pluggy==1.6.0
     + puremagic==1.30
     + pydantic==2.11.7
     + pydantic-core==2.33.2
     + python-dateutil==2.9.0.post0
     + python-ulid==3.1.0
     + pyyaml==6.0.2
     + setuptools==80.9.0
     + six==1.17.0
     + sniffio==1.3.1
     + sqlite-fts4==1.0.3
     + sqlite-migrate==0.1b0
     + sqlite-utils==3.38
     + tabulate==0.9.0
     + tqdm==4.67.1
     + typing-extensions==4.15.0
     + typing-inspection==0.4.1
    Installed 1 executable: llm

    zeta:~ blyth$ which llm
    /Users/blyth/.local/bin/llm
    zeta:~ blyth$ 


Upgrade llm with uv
---------------------

::

    uv tool upgrade llm


Plugins
---------

* https://llm.datasette.io/en/stable/plugins/directory.html#plugin-directory


* https://github.com/simonw/llm-gguf

* https://github.com/simonw/llm-mlx

* https://github.com/taketwo/llm-ollama

* https://github.com/simonw/llm-mlc
* https://llm.mlc.ai/docs/get_started/introduction.html#introduction-to-mlc-llm


llm-ollama
-----------

::

    zeta:~ blyth$ ollama serve
    Error: listen tcp 127.0.0.1:11434: bind: address already in use
    zeta:~ blyth$ llm ollama models
    model       digest          capabilities               
    qwen3:8b    500a1f067a9f    completion, tools, thinking
    zeta:~ blyth$ 

    zeta:~ blyth$ llm -m qwen3:8b "How much is 2+2 ?"










EOU
}
llm-get(){
   local dir=$(dirname $(llm-dir)) &&  mkdir -p $dir && cd $dir

}
