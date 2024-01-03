#!/bin/bash

# Download the files into the "data" folder
wget -P ./raw_data https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowAll.csv
wget -P ./raw_data https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowSep.csv