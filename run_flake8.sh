#! /usr/bin/env bash
#
# this script runs flake8 on all clean files.

set -e
echo "Executing flake8..."


Color_Off='\033[0m'
Red='\033[0;31m'
Green='\033[0;32m'

function fail {
    echo -e "${Red}FLAKE8 FAILED!!!$Color_Off"
    exit 1
}

flake8 \
    --count \
    --show-source \
    --statistics \
    ./test/ \
    ./setup.py \
    ./two4two/ \
    || fail

echo -e "${Green}PASSED$Color_Off"
