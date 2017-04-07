#!/bin/bash

# subjects from http://api.elsevier.com/content/subject/scidir?httpAccept=text/xml
subjects=(11  # computer science
          19  # materials science
          24) # physics and astronomy
year_begin=2010
year_end=2016

for subject in ${subjects[@]}; do
    for ((year=${year_begin}; year<=${year_end}; year++)); do
        python crawl.py ${subject} ${year}
    done
done
