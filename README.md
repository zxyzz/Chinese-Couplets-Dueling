# Master Semester Project

## Dueling Chinese Couplet using Sequence to Sequence Model

### Pre-requisites
To run the project, you will need python3 along with the libraries: 
- `numpy`
- `matplotlib`
- `torch`<br/>
The codes are tested on `Python 3.7.3`.

### Datasets
- `vocab.txt` contains all vocabularies extracted from couplet datasets.
- `in.txt` contains couplet inputs for the training process.
- `out.txt` contains desired couplet outputs for the training process.
- `test_in.txt` contains couplet inputs for the evaluation process.
- `test_out.txt` contains desired couplet outputs for the evaluation process.


### Structure of the codes
This repo contains several files for training and evaluating processes. 

##### - run_training.py
This file is used to train the sequence to sequence model. The trained encoder and decoder models will be save as `pt` files placed in `models` folder.
The following command was used for training in this project.
```
python run_training.py -a True -ep 204
```
Default values are used for other hyperparameters. They can be tuned optionally. Please refer to *help* for further information.

##### - evaluation.ipynb
The obatined encoder and decoder models from `run_training.py` can be used in `evaluation.ipynb` in order to evaluate the performance of the model. If run `evaluateRandomly` method, the evaluation results can be displayed in the notebook and will also be saved locally in `couplet_outputs.txt` which contains the index of couplet in the `test_in.txt`, and the corresponding couplet input, the model answer and the desired answer (ground-truth). Further more, the attention plots of couplets are likewise displayed in notebook and saved as `png` files locally.

##### - encoderRNN.py
This file contains codes for encoder constructed using bidirectional GRU.
##### - attention_II.py
This file contains codes for attention mechanism: `general` or `dot` alternatives.
##### - attnDecoderRNN_II.py
This file contains codes for decoder constructed using bidirectional GRU with attention mechanism.
##### - utils.py
This file contains helper functions.
##### - models
This folder contains the trained encoder and decoder models.
- II_enc_epoch_204.pt
This is the encoder model trained for 204 epochs.
- II_dec_epoch_204.pt
This is the decoder model trained for 204 epochs.

### Resources
- Seq2seq trainslation tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
- Attention: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
- Chinese Couplet dataset: https://github.com/wb14123/couplet-dataset
