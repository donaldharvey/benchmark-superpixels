Superpixel benchmarking
=======================

## Usage

`./benchmark-superpixels -B [path-to-BSDS-dataset] [options] [algorithm-or-input-dat-directory] [output-file]`

Run benchmarks on a superpixel algorithm, either generating the segmentations on-the-fly or from a saved dataset, and outputs to `[output-file]` in TSV format.

If `algorithm-or-input-dat-directory` is `slic` or `seeds`, use that algorithm to segment each image in the BSDS500 dataset.
Otherwise, it looks in that directory for `.dat` files, which should be named `[BSDS500 basename].dat` (e.g. `100007.dat`)

For on-the-fly generation, target number of superpixels may be specified with `-N` or `--number-superpixels`.

## Examples

    - run SLIC on the fly 
    `./benchmark-superpixels -B /Downloads/BSDS500 -N 2048 slic SLIC-1000.tsv`

    - use a set of saved segmentations
    `./benchmark-superpixels -B /Downloads/BSDS500 /path/to/saved/segmentations output.tsv`