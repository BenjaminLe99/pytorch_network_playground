
#!/bin/bash

pyenv_activate ()
{
    export PYENV_ROOT="/afs/desy.de/user/w/wiedersb/.pyenv/";
    export PATH="${PYENV_ROOT}/bin/:${PATH}";
    eval "$(pyenv init --path)";
    eval "$(pyenv init -)";
    eval "$(pyenv virtualenv-init -)"
}

setup_env() {
    # Prevent double execution
    if [ -n "$SETUP_COMPLETE" ]; then
        echo "Setup already completed. Skipping."
        return 0
    fi
    # Get the directory where this script is located
    export LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Source config.sh from the same directory
    source "$LOCAL_DIR/config.sh"

    # Activate pyenv environment stored in ML_ENV
    if [ -n "$ML_ENV" ]; then
        # check if pyenv exists if not source before
        pyenv activate "$ML_ENV"
        echo "activate virtualenv ${ML_ENV}"
        export PS1="[${VIRTUAL_ENV##*/}] $PS1"

    else
        echo "ML_ENV is not set."
        return 1
    fi
    # Show only the last part of the virtualenv path (env name) in the prompt
    export PYTHONPATH=$(pwd)/src
    echo "Ready to go"
    export SETUP_COMPLETE=1
}

# entry point
if ! command -v pyenv >/dev/null 2>&1; then
    pyenv_activate
fi

setup_env "$@"
