#!/bin/bash

if [ -v CODE_PATH ]; then
    echo "Installing $CODE_PATH inplace."
    pip install -e $CODE_PATH
fi

/bin/bash

