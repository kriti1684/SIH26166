#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ch2_isis_dev

echo "Checking ISISDATA..."
echo "ISISDATA=$ISISDATA"

echo "Downloading CK DB files..."
downloadIsisData chandrayaan2 "$ISISDATA" --include="kernels/ck/*.db" --include="kernels/ck/*.conf" --exclude="kernels/ck/**"

echo "Done."
