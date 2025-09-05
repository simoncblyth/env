# === func-gen- : tools/hf fgp tools/hf.bash fgn hf fgh tools src base/func.bash
hf-source(){   echo ${BASH_SOURCE} ; }
hf-edir(){ echo $(dirname $(hf-source)) ; }
hf-ecd(){  cd $(hf-edir); }
hf-dir(){  echo $LOCAL_BASE/env/tools/hf ; }
hf-cd(){   cd $(hf-dir); }
hf-vi(){   vi $(hf-source) ; }
hf-env(){  elocal- ; }
hf-usage(){ cat << EOU

Within each project venv::


    uv pip install -U "huggingface_hub[cli]"
    uv pip install -U "requests[socks]"

    ALL_PROXY=socks5h://127.0.0.1:8080 hf auth login --token $HF_TOKEN --add-to-git-credential


    (llama_cpp_python) A[blyth@localhost llama_cpp_python]$ ALL_PROXY=socks5h://127.0.0.1:8080 hf auth login --token $HF_TOKEN --add-to-git-credential
    Token is valid (permission: read).
    The token `HF_Read` has been saved to /home/blyth/.cache/huggingface/stored_tokens
    Cannot authenticate through git-credential as no helper is defined on your machine.
    You might have to re-authenticate when pushing to the Hugging Face Hub.
    Run the following command in your terminal in case you want to set the 'store' credential helper as default.

    git config --global credential.helper store

    Read https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details.
    Token has not been saved to git credential helper.
    Your token has been saved to /home/blyth/.cache/huggingface/token
    Login successful.
    Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.
    (llama_cpp_python) A[blyth@localhost llama_cpp_python]$ 







EOU
}
hf-get(){
   local dir=$(dirname $(hf-dir)) &&  mkdir -p $dir && cd $dir

}
