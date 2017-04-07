#!/bin/sh

# Copyright 2016 Tom Kenter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ -z "$1" ]; then
    echo
    echo " USAGE: $0 PPDB_FILE"
    echo
    echo " This script reads a PPDB file, and outputs a vocabulary file"
    echo " suitable to be used by Siamese CBOW".
    echo
    echo " Output is to stdout"
    echo
    exit
fi

PPDB_FILE=$1

python ppdbUtils.py -single_sentence $1 | cut -f2 | tr ' ' '\n' | sort | uniq -c | sort -nr 
