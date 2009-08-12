#!/bin/bash
# thho_profile.sh

export thhoenv=$HOME/env/thho/profile/pdsf

if [ -e $thhoenv/tty_profile.sh ]; then
    echo "source $thhoenv/tty_profile.sh"
    source $thhoenv/tty_profile.sh
fi

if [ -e $thhoenv/module_loading.sh ]; then
    echo "source $thhoenv/module_loading.sh"
    source $thhoenv/module_loading.sh
fi
