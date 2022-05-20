#!/bin/bash

if [ -d "venv" ]; then
    echo "VENV already exists, will be removed"
    rm -rf venv
fi

echo "VENV will be created" && \
python3 -m venv venv && \
source venv/bin/activate && \
echo "Install requirements..." && \
pip3 install -r requirements.txt && \
echo "Requirements have been successfully installed" && \
deactivate
