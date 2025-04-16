#!/bin/bash
# Usage: ./generate_GENE_BED.sh input.gtf[.gz] output.tsv

# Check for correct number of arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 input.gtf[.gz]"
    exit 1
fi

input_file="$1"
output_file="${input_file}.gene.bed"

# awk function
process_file() {
    awk -F "\t" 'BEGIN { OFS="\t" }
    # 3rd column contains annotation
    $3 == "exon" {
        gene_name = ""
        # Split the 9th column (attributes) by semicolon
        split($9, attrs, ";")
        for (i in attrs) {
            gsub(/^[ \t]+|[ \t]+$/, "", attrs[i])
            # Check if the attribute contains "gene_name"
            if (attrs[i] ~ /^gene_name=/) {
                split(attrs[i], keyval, "=")
                gene_name = keyval[2]
                break
            }
        }
        if (gene_name != "") {
            print $1, $4, $5, gene_name
        }
    }'
}

if file "$input_file" | grep -q 'gzip compressed data'; then
    zcat "$input_file" | process_file > "$output_file"
else
    process_file "$input_file" > "$output_file"
fi

echo "TSV file created: $output_file"
