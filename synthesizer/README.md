## About

In this folder you can find all the files needed for the synthesis process:
* `synthesizer.py` contains the main code.
* `utils` contains several files with functions needed by the synthesizer, but which could be reused for other purposes.
* `data` contains data necessary to run the synthesizer, namely background images, corpora and a list of usable fonts.
* `synthesize` is a wrapper script which calls the required functions from `synthesizer.py`

## Usage

### Font setup
To use the synthesizer, you first need to set up the fonts. The included font list in the `data` folder references about 400 handwritten fonts which you can download [here](https://drive.google.com/file/d/1fd9qU3OtYWGMT4PYrcLzYh-I2rlQfqMw/view?usp=sharing). You must install those fonts in your system for the synthesizer to be able to use them. If you want to add more fonts, you must append a valid alias in the font list. The method for acquiring a font alias might vary depending on your system. If you're using linux, it's likely that the command `fc-scan <folder name> | grep fullname:` will work to get the alias of all fonts inside a folder.

### Calling `synthesize`

The easiest way to use the synthesizer is to call `synthesize` from the command line.
Below is a description of all the arguments you can pass.

Global parameters:
* `-font-size=N` sets the font size to N pixels of height. This guarantees each line will have exactly this height independently of the font, but the width can vary.
* `-words` if present, word segments will be generated along with their groundtruth.
* `-lines` if present, line segments will be generated along with their groundtruth.
* `-const-width=N` if present, the synthesizer will try to make each letter of the segments have, in average, N pixels of width.
* `-distort-bboxes` if present, the bounding boxes of segments will be slightly distorted randomly. Note that enabling or disabling this feature will interfere with the RNG.
* `-clone-dataset=PATH` if present, the synthesizer will generate a clone of a dataset. PATH must indicate the path to the data necessary to perform the clone. For more information about this data, see the `IAM_utilities` folder at the top level of the repository.
* `-out-dir=PATH` specifies the output directory.
* `-load-last-seed` if present, the seed used for RNG on the last execution will be loaded, thus producing the same output.

Parameters only relevant when not cloning a dataset:
* `-num-pages=N` specifies how many pages of text should be generated.
* `-page-width=N` specifies the page width in pixels. If not present, the page width is set to 30 times the font size.
* `-chars-per-page=N` specifies how many characters will be pulled from the corpus for each page.
* `-corpus=PATH` if present, the specified path will be used as the corpus. Otherwise, the default online corpus will be used.

Misc:
* `-list-fonts` lists all available fonts able to be used with the synthesizer.
* `-help` lists all the accepted arguments, their type and their default value.

#### Call examples

Generate 10 pages of text feeding from the default corpus. Generate only forms and words. Distort the bounding boxes of the words.
```bash
./synthesize -num-pages=10 -words -distort-bboxes
```
Clone the IAM dataset (data previously generated and saved in `IAM_data`), generating all forms, lines and words.
```bash
./synthesize -clone-dataset=IAM_data -words -lines
```

### Modifying distortions and other parameters

For now, to modify the distortion pipeline, you must manually change it in the source code. You can find it in the function `generate_page` of the `Synthesizer` class.
There are other modifiable parameters for which there is yet no interface, see the comments in the source code for more information on how to modify them.

