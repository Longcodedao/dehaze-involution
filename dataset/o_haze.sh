#!/bin/bash

# Define the URL and the output directory
URL="http://www.vision.ee.ethz.ch/ntire18/o-haze/O-HAZE.zip"
OUTPUT_DIR="O-HAZE"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Download the dataset
echo "Downloading the dataset..."
wget -O O-HAZE.zip $URL

# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download the dataset. Exiting."
    exit 1
fi

# Extract the dataset
echo "Extracting the dataset..."
unzip O-HAZE.zip -d $OUTPUT_DIR

# Check if the extraction was successful
if [ $? -ne 0 ]; then
    echo "Failed to extract the dataset. Exiting."
    exit 1
fi

# Clean up the zip file
echo "Cleaning up..."
rm O-HAZE.zip

echo "Download and extraction complete. The dataset is available in the '$OUTPUT_DIR' directory."
