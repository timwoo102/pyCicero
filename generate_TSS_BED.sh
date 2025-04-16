#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 input_gtf.gz [output_bed]"
    exit 1
fi

# Input GTF file
INPUT_GTF="$1"

# Define output BED file (default: replace .gtf.gz with .tss.bed)
if [ "$#" -ge 2 ]; then
    OUTPUT_BED="$2"
else
    OUTPUT_BED="${INPUT_GTF%.gtf.gz}.tss.bed"
fi

echo "Processing input GTF file: $INPUT_GTF"
echo "Output BED file will be: $OUTPUT_BED"

# Extract TSS and output as BED:
zcat "$INPUT_GTF" | awk 'BEGIN {OFS="\t"} $3=="transcript" {
    if ($7 == "+") {
        print $1, $4
    } else if ($7 == "-") {
        print $1, $5
    }
}' | sort -k1,1V -k2,2n > "$OUTPUT_BED"

if [ $? -eq 0 ]; then
    echo "TSS BED file successfully generated: $OUTPUT_BED"
else
    echo "Error: Unable to generate TSS BED file."
    exit 1
fi