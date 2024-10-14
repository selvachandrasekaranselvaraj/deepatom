 grep -e '!' *out | awk '{print $5 "\t  " $1}' | cut -c1-23 | awk '{print $2 "\t  " $1}' > energy.txt
