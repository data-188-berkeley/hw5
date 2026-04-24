#!/bin/bash
# Prepare submission.
# Usage:
#   ./prepare_submission.sh <ZIP_OUTPATH>
# Example:
#   ./prepare_submission.sh hw0.zip

set -e

zip_outpath=$1

if [ -f "${zip_outpath}" ]; then
    # if file exists, delete it
    rm -f "${zip_outpath}"
fi

# zip entire current working directory, with excludes to keep filesize small
# HW5: exclude autograder_student.pt, test_reference.pt (they contain large model weights that would increase submission zipfile size to ~85MB!)
zip -r "${zip_outpath}" . -x "*.git*" "data/*" "*.ipynb_checkpoints*" ".env/*" "*.pyc" "*.pytest_cache*" "*__pycache__*" "autograder_student.pt" "test_reference.pt"
echo "Created zip at: ${zip_outpath}"
