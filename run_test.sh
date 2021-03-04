#! /usr/bin/env bash
#
# this script should pass before committing. It executes flake8 and pytest
# and has a non-zero exit code if one fails.
#
# The script will forward any command line arguments to pytest, e.g.:
#
#   $ ./run_test.sh -k test_blender_rending
#
# will only test the ``test_blender_rending`` function.
#


Color_Off='\033[0m'
Red='\033[0;31m'
Green='\033[0;32m'

function fail {
    echo -e "${Red}FLAKE8 Sanity FAILED!!!$Color_Off"
    exit 1
}


# Early stop the test if there are Python syntax errors or undefined names.
echo "Executing flake sanity checks..."
flake8 ./two4two \
       ./test  \
       ./setup.py  \
       ./examples \
       --select=E9,F63,F7,F82 \
       --show-source \
       --statistics || \
       fail

echo "Executing pytest..."
python -m pytest -v -s test/  \
    -m "not slow" \
    "$@"

pytest_ret=$?

./run_flake8.sh
flake_ret=$?

# if one is non-zero the sum is non-zero
exit_code=$((flake_ret + pytest_ret))

Color_Off='\033[0m'
Red='\033[0;31m'
Green='\033[0;32m'

if [[ "$exit_code" != "0" ]]; then
    echo -e "${Red}FAILED :($Color_Off"
    exit 1
else
    echo -e "${Green}PASSED :)$Color_Off"
fi
