## About this fork

This fork focuses on improving the handwritten document synthetization aspect of the original repo.
I tried to keep the synthesizer as stand-alone as possible, so some files and code were relocated or removed.

These are the main modifications which were made (based on the version cloned on Apr 28, 2018):
- Added new distortions: elastic deformation and ink degradation.
- Modified existing distortions: spline interpolation, document noise, background blending.
- Added the option to disable distortions while still generating parameters (useful when using a fixed seed for the RNG).
- Added more options to text rendering (see comments in the code).
- Words and lines are now split using the text string and not the character bounding boxes (much more reliable).
- The synth can now generate a synthetic clone of a dataset if the necessary data is provided.
- The synth can now generate pages, lines and words at the same time.
- Misc bug fixes and code refactoring.

<sup>Note: this list is not a comprehensive changelog.</sup>

### Known issues
There are two main known issues which need to be solved:
- When using UTF-8 text, words and lines get correctly segmented at string level, but not at image level.
- The function `raw_interp2` found in `utils/image_processing.py` wrongly rotates the image by 180º.

## Install dependencies
To run this code, you must have Python 2.7 installed.
The following Python packages are also required:
```bash
pip install --user --upgrade Pillow matplotlib numpy opencv_contrib_python pycairo scikit_image scipy
```

## Usage

See the readme in the `synthesizer` and `IAM_utilities` folders for usage information.

