
Code for the paper [Exploring Graph Representations of Logical Forms for Language Modeling](https://arxiv.org/pdf/2505.14523) (and [Chapters 4-6 of my dissertation](https://mjs227.github.io/home/dissertation.pdf)).

# Installation and Setup:
1. Create a new virtual environment with python==3.11.7
2. Clone the repository
3. Install the dependencies:
   
   ```
   pip install -r requirements.txt
   ```
4. Run setup.py, where GLOBAL_FILEPATH={PATH_TO_DIR}/{DIR} is the directory that will store all of the data for pretraining and experiments. PATH_TO_DIR should exist already, but DIR should not (it will be created by setup.py):
   ```
   python3 setup.py -g GLOBAL_FILEPATH
   ```
5. Install the ACE-ERG parser:
   - Download ace-0.9.34-x86-64.tar.gz and erg-1214-x86-64-0.9.34.dat.bz2 from [https://sweaglesw.org/linguistics/ace/](https://sweaglesw.org/linguistics/ace/)
   - Extract the directory ace-0.9.34 from ace-0.9.34-x86-64.tar.gz and place it into GLOBAL_FILEPATH/
   - Extract erg-1214-x86-64-0.9.34.dat from erg-1214-x86-64-0.9.34.dat.bz2 and place it into GLOBAL_FILEPATH/ace-0.9.34/

# Pretraining:

## Setup:
1. Run PROJECT_DIRECTORY/preprocessing/setup_wiki_data.py (this will take a while)
   - Use the -k argument to select the number of sentences to retain from the dataset (for GFoLDS and BERT MLM; BERT NSP data is filtered in randomize_batches.py below).
2. Run PROJECT_DIRECTORY/preprocessing/create_pretraining_data.py. WARNING: this will take a *really* long time (likely a month or more). Use more threads (the --num_workers argument) if you have them!
   - Note: this script is also used to create pretraining data for the BERT comparison models. See the --bert and --nsp flags.
3. Run PROJECT_DIRECTORY/preprocessing/randomize_batches.py. This will pre-shuffle the pretraining batches, so they don't need to be shuffled during pretraining. If you don't want to do that, set the --num_epochs argument to 1 (default), and you will need to set RANDOMIZE_BATCHES=True during pretraining (see below). Otherwise, set --num_epochs to the number of planned pretraining epochs.
   - Note: when using this script to create NSP pretraining data for BERT (--nsp flag), use the -k argument to set the number of sentence pairs that are retained (the rest will be discarded).

## Training:
1. Configure hyperparameters in PROJECT_DIRECTORY/pretraining/gfolds/config.py (or PROJECT_DIRECTORY/pretraining/bert/config.py, depending on the model). If you set num_epochs=1 when running randomize_batches.py, set RANDOMIZE_BATCHES=True here. Otherwise, set RANDOMIZE_BATCHES=False.
2. Run the training script(s):
   - GFoLDS: PROJECT_DIRECTORY/pretraining/gfolds/train.py
   - BERT (MLM): PROJECT_DIRECTORY/pretraining/bert/train.py
   - BERT (MLM+NSP): PROJECT_DIRECTORY/pretraining/bert/train_nsp.py
  
# Experiments:

Note: most of these experiments cannot be run without first pretraining the model(s), or at least running the pretraining setup scripts.

## SNLI:
1. Download and setup the dataset (this only needs to be done once):
   - Download snli_1.0.zip from [https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)
   - Extract the directory snli_1.0 from snli_1.0.zip and place it into GLOBAL_FILEPATH/nli/data/
   - Run PROJECT_DIRECTORY/experiments/SNLI/create_data.py (this will take a while)
2. Run the script(s):
   - GFoLDS: PROJECT_DIRECTORY/experiments/SNLI/gfolds.py
   - BERT: PROJECT_DIRECTORY/experiments/SNLI/bert.py

## RELPRON:
1. Run PROJECT_DIRECTORY/experiments/RELPRON/create_data.py
2. Run PROJECT_DIRECTORY/experiments/RELPRON/evaluate.py

## MegaVeridicality V2.1:
1. Download and setup the dataset (this only needs to be done once):
   - Download mega-veridicality-v2.1.zip from [https://megaattitude.io/projects/mega-veridicality/](https://megaattitude.io/projects/mega-veridicality/)
   - Extract mega-veridicality-v2.1.tsv from mega-veridicality-v2.1.zip and place it into GLOBAL_FILEPATH/factuality/data/
   - Run PROJECT_DIRECTORY/experiments/MegaVeridicality/create_data.py
2. Run PROJECT_DIRECTORY/experiments/MegaVeridicality/evaluate.py

## McRae et al.:
1. Run PROJECT_DIRECTORY/experiments/McRae_etal/get_embeddings.py
2. Run PROJECT_DIRECTORY/experiments/McRae_etal/evaluate.py

## ALH (a.k.a. LKCH, depending on the paper version):
1. Run PROJECT_DIRECTORY/experiments/ALH/create_data.py
2. Run PROJECT_DIRECTORY/experiments/ALH/run_elem_probes.py
   - Note: this will only run the elementary task evaluation. For the RELPRON evaluation, you will need to use PROJECT_DIRECTORY/experiments/RELPRON/evaluate.py with the --split argument set to "dev".

   
