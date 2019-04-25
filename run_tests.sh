#!/bin/bash
RUN_ONLY_TEST=${1}
if [ ! -z $RUN_ONLY_TEST ]
then
  echo "RUN TEST $RUN_ONLY_TEST"
  python -m unittest $RUN_ONLY_TEST
else
  echo "RUN ALL TESTS"
  python -m unittest discover -v tests
fi
