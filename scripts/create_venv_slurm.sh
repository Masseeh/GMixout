#!/bin/bash

start=`date +%s`

module purge
module load python/3.10

## skip venv creation if already exists
if [ ! -d ".venv" ]; then
    start=`date +%s`
    echo "creating virtual environment in $PWD/.venv"

    virtualenv --no-download .venv
    source .venv/bin/activate

    echo "installing requirements in virtual environment"

    pip install --no-index -r requirements.txt
    end=`date +%s`
    echo "virtual environment created and requirements installed in $((end - start)) seconds"
else
    echo "virtual environment already exists, skipping creation"
fi

end=`date +%s`
echo "Virtual environment setup completed in $((end - start)) seconds"