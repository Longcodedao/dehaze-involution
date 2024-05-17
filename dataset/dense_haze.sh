#!/bin/bash

# Define the URL and the output directory
URL="https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/files/Dense_Haze_NTIRE19.zip"
OUTPUT_DIR="Dense-HAZE"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Download the dataset
echo "Downloading the dataset..."
wget -O Dense-HAZE.zip $URL

# Check if the download was successful
if [ $? -ne 0 ]; then
    echo "Failed to download the dataset. Exiting."
    exit 1
fi

# Extract the dataset
echo "Extracting the dataset..."
unzip Dense-HAZE.zip -d $OUTPUT_DIR

# Check if the extraction was successful
if [ $? -ne 0 ]; then
    echo "Failed to extract the dataset. Exiting."
    exit 1
fi

# Clean up the zip file
echo "Cleaning up..."
rm Dense-HAZE.zip

echo "Download and extraction complete. The dataset is available in the '$OUTPUT_DIR' directory."