#!/bin/bash

# Install unzip if not already installed
sudo apt update && sudo apt install -y unzip

# URL of the files to be downloaded
URL1="https://www.dropbox.com/scl/fo/445j1h6nswvto807ipkon/AJln0vgFNYXaM7ogwEMO55s/train?rlkey=was6y4tqd3adjggohjts2mqw1&dl=1"
URL2="https://www.dropbox.com/scl/fo/445j1h6nswvto807ipkon/AB02TocZAguS8oPQo5xNXL8/test?rlkey=was6y4tqd3adjggohjts2mqw1&dl=1"

# Target directory for download
DOWNLOAD_DIR="foggy_cityscape"

# Create download directory if it does not exist
mkdir -p "${DOWNLOAD_DIR}"

# Filename extraction from URL
FILENAME1=$(basename "${URL1}")
FILENAME2=$(basename "${URL2}")

# Full path to the downloaded files
FILEPATH1="${DOWNLOAD_DIR}/${FILENAME1}"
FILEPATH2="${DOWNLOAD_DIR}/${FILENAME2}"

# Download the files using wget
wget -c "${URL1}" -O "${FILEPATH1}"
if [ $? -ne 0 ]; then
    echo "Download of file 1 failed."
    exit 1
fi

wget -c "${URL2}" -O "${FILEPATH2}"
if [ $? -ne 0 ]; then
    echo "Download of file 2 failed."
    exit 1
fi

TRAIN_DIR="train"
TEST_DIR="test"

mkdir -p "${TRAIN_DIR}"
mkdir -p "${TEST_DIR}"

# Unzip the downloaded files
unzip -o "${FILEPATH1}" -d "${DOWNLOAD_DIR}/${TRAIN_DIR}"
unzip -o "${FILEPATH2}" -d "${DOWNLOAD_DIR}/${TEST_DIR}"

rm -rf "train"
rm -rf "test"

echo "Process completed. Check ${DOWNLOAD_DIR} for the files."
