This is a repository containing the code used for the MSc Thesis 
**Mind the Linguistic Gap: Studying the learning of linguistic properties
of continuous sign language videos in a Sign Classification task**
and for the paper **Exploring the Importance of Sign Language Phonology for a 
Deep Neural Network** submitted at [ESANN 2023](https://www.esann.org/).

# Exploring the Importance of Sign Language Phonology for a Deep Neural Network

---

### Authors: Javier Martinez Rodríguez, Martha Larson, and Louis Ten Bosch

### 05-2023

---

Note that scripts were ran on a ssh connection through the use of shell scripts, available
`shell` folder. Scripts can also be ran locally.

## Data Setup

- Please make sure to use `requirements.txt` file to install all package dependencies.
- Datasets [Corpus NGT](https://www.corpusngt.nl/) and [NGT Signbank](https://signbank.cls.ru.nl/) are private are
thus not included in this repository. Contact the authors to gain access.
- Create a folder and include both datasets
  1. Corpus NGT videos with format `CNGTXXXX_SYYY.mpg` and their corresponding annotations with format 
  `CNGTXXXX.eaf`. `XXXX` denotes the number of the video, while `YYY` denotes the signer appearing in the video. 
  Note that the later can be manually read by downloading the [ELAN software](https://archive.mpi.nl/tla/elan/download) 
  2. NGT Signbank videos with format `GLOSS-ID.mp4`
- Download a file named by navigating to [NGT Signbank](https://signbank.cls.ru.nl/), selecting the dropdown
`Signs` > `Show all signs`, and clicking on the `CSV` button. This downloads a file `dictionary-export.csv`
includes the correspondence of glosses with their IDs, as well as the list of phonological
(and other) aspects used in this work.

## Pre-processing

Be sure to run the pre-processing scripts in the following order.

1. Run the script `get_signbank_vocab_from_csv.py` that will save a `.gzip` file with the gloss-id mappings
for the Corpus NGT.
2. Run the script `split_cngt.py` that will split the annotations of both signers.
3. Run the script `resize_videos.py` where you should specify the desired `video_size` and the `framerate`.
4. Run the script `extract_isolated_videos_cngt.py` to obtain the final data where isolated signs are
extracted based on the gloss annotations. The parameter `window_size` that determines the number of frames
per video must be set, as the window gets filled by looping the video.

## Training

During training, we use early stopping with ``patience=10`` and ``min_delta=0.0``. We also perform learning rate reduction
on plateau with ``patience=5`` with a factor of `0.1`. Both callbacks monitor the validation loss. 
The minimized loss is the binary cross-entropy, and weights checkpoints are only stored when validation loss is reduced.

To train the I3D model, run the script `train_i3d.py`. Note that hyperparameters must be previously
set in the `train_config.yaml` file found in the `config` folder. The training history will be saved
in both a `.png` file and a `.txt` file in the model checkpoints folder.

The signs used in the binary classification are set using the arguments ``reference_sign`` and `train_signs`.
Note that the latter is a list, and therefore any number of signs can be introduced to train different 
classifiers with the signs specified in the list against the reference sign, as done in the 
**phonological distance experiment.**

### Hyperparameters

The training hyperparameters used in the paper are:
- epochs = 50
- batch size = 128
- initial learning rate = 0.1
- SGD momentum = 0.9
- weight decay = 1e-7

Furthermore, there are some data parameters that must be set too:
- `clips_per_class` determines the number of samples loaded for training. This was used for
the **phonological distance experiment**. If `-1` is used, all
samples are used.
- `window_size` and `input_size` must match the window size and video size specified 
during the pre-processing step.
- `loading_mode` selects the way the data is loaded. This argument can be set to:
  - `'random'` to randomly split the data.
  - `'balanced'` to keep class imbalance through train/val/test splits. **This is the value used in this work.**
  - `'stratified'` to load different signers (with some overlap to keep 4:1:1 ratio) in the 
  train/val/test splits. Note that this slightly alters the 4:1:1 split. 

## Predicting

Predictions can be ran in any data split by using the script ``predict_i3d.py`` and by setting the 
parameters in the ``predict_config.yaml`` file. The parameters are similar to the `train_config.yaml` file,
with the difference that the run name, batch size, initial learning rate, and number of epochs of the training
step must be specified in order to load the corresponding model weights. 

The last model checkpoint is loaded by default. Predictions can be stored by setting ``save_predictions=True`` and are divided
automatically into TP, TN, FP, and FN. The confusion matrix and the prediction metrics are saved too.

## Intrinsic dimension estimation

Similarly to the previous training and prediction steps, the intrinsic dimension can be estimated by 
running the ``get_intrinsic_dim.py`` script and by specifying the model parameters in the `get_id_config.yaml` file.

Note that the ID is calculated on the feature vector of the last hidden layer. To observe the ID of the representations
learnt during training, be sure to specify the use of the training split.

## Other functionalities

Other useful scripts can be found in the ``utils`` folder, namely:

- ``parse_cngt_glosses.py`` contains the function that parses glosses. This unifies and cleans glosses
that are not useful, and discards glosses that are too complex for the classification.
- ``determine_ling_dist.py`` calculates all pair-wise linguistic distances between glosses and prints
the most frequent reference signs and their most frequent signs 10 comparison signs that can be found in
Table 2.
- ``get_cngt_gloss_frequency.py`` gets the total number of gloss annotations in the Corpus NGT and the
average of annotations per gloss.
- ``get_frame_length_distribution_from_anns.py`` outputs a distribution plot that shows distribution
of number of frames per sign in the Corpus NGT.
- ``get_signer_stats_from_anns.py`` outputs a plot showing the number of annotated glosses per signer
in the Corpus NGT.
- ``get_split_frame_distributions.py`` outputs a plot that shows the number of frames per signs for
the train/val/test splits.
- ``visualize_clips_durations.py`` outputs a distribution plot of the sign length (in seconds) for 
the Corpus NGT and NGT Signbank.



