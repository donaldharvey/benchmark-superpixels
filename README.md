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
    ```bash
    ./benchmark-superpixels -B /Downloads/BSDS500 -N 2048 slic SLIC-1000.tsv
    ```

- use a set of saved segmentations  
    ```bash 
    ./benchmark-superpixels -B /Downloads/BSDS500 /path/to/saved/segmentations output.tsv
    ```

## Segmentation format specification (.dat files)

Segmentation data should be in the following form:

```
[int32 width]
[int32 height]
(array of int32s, starting at 1, identifying each pixel with a superpixel label)
(array of uchars with values 0 or 1, where 1 represents a boundary pixel)
```

Example output code, where `boundary_data` and `segmentation_data` are OpenCV `Mat` instances with boundary mask and label data respectively:
```c++
uchar* contoursOut = NULL;
int32_t* regionsOut = NULL;

contoursOut = new uchar[image.cols * image.rows];
regionsOut = new int32_t[image.cols * image.rows];

for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
        contoursOut[i*image.cols + j] = boundary_data.at<uchar>(i, j);
        regionsOut[i*image.cols + j] = int32_t(segmentation_data.at<int>(i, j));
    }
}

ofstream outfile;
outfile.open("seg.dat", ios::binary | ios::out);

outFile.write((char*)&(image.cols), sizeof(int32_t));
outFile.write((char*)&(image.rows), sizeof(int32_t));

for (int i=0; i < image.cols*image.rows; i++) {
    outFile.write((char*)&(regionsOut[i]), sizeof(int32_t));
}

for (int i=0; i < image.cols*image.rows; ++i) {
    outFile.write((char*)&(contoursOut[i]), sizeof(uchar));
}
outFile.close();
```

The BSDS data should also be in this form, with filenames `[name]_[x].dat`, where `name` is the image name and `x` is the number corresponding to which ground truth version of the image is used. A Python script is provided to generate these from the .mat files; they are also available to download [here](https://drive.google.com/file/d/0B5aUH9uZcfGOb3pkVGJncnBvVEk/view?usp=sharing). (Careful - it's abut 2GB uncompressed!)
