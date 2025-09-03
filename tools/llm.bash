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

* https://simonwillison.net/2025/May/2/qwen3-8b/

* https://medium.com/@billcava/terminal-ai-how-llm-changed-my-workflow-71ef97ddab5b









EOU
}
llm-get(){
   local dir=$(dirname $(llm-dir)) &&  mkdir -p $dir && cd $dir

}
