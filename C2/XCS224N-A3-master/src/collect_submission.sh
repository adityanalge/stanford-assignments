#!/bin/bash

# Delete existing zip file if it exists
rm -f assignment3_submission.zip

# Change into the submission directory
cd submission || exit
echo "Collecting submission files..."

# Use PowerShell to zip files (works on Windows natively)
powershell.exe -Command "Compress-Archive -Path '__init__.py','parser_model.py','parser_transitions.py','parser_utils.py','train.py' -DestinationPath '../assignment3_submission.zip'"

# Go back to original directory
cd ..

echo "Done!"
