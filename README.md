## prepare python environment

```commandline
pip install -r requirements.txt
```

## Common parameter setting

`common_settings/*.sh` is the dataset specific parameters, such as max query length, etc.

## Multi-Vector Denser Retrieval (MVDR)

`mvdr.py` and `faiss_index.py`

reproduce `ColBERT`, based on `T5`, you can change it to `BERT`

run `bash mvdr_run_scripts/run.sh`

This will create an output diretory in `output/mvdr.DATA.train` by default, which is the `$EXP_DIR` defined in `mvdr_run_scripts/params.sh`

All MVDR related parameters are in `mvdr_run_scripts/params.sh`.

## Generative Retrieval (GR)

### install SEAL

ref. [facebookresearch](github.com/facebookresearch/SEAL)

SEAL needs a working installation of [SWIG](https://www.swig.org/), e.g. (on Ubuntu):
```commandline
sudo apt install swig
```

or using `conda`:

```commandline
conda install swig
```

We also assume that `pytorch` is already available in your environment. SEAL has been tested with version 1.11.

Clone this repo with `--recursive` so that you also include the submodule in `res/external`.
```commandline
git clone --recursive https://github.com/facebookresearch/SEAL.git
```

Compile and install `sdsl-lite`:
```commandline
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
```

Install other dependencies:
```commandline
pip install -r requirements.txt

# pyserini
# pip install -r requirements_extra.txt
```

Now install this library.
```commandline
pip install -e .
```

#### Note

Since SEAL uses an older version of transformers, so there are a few places that need to be updated

You can directly copy our fixed version of `seal/beam_search.py` to your SEAL workplace

```commandline
cp beam_search.py /path/to/SEAL/seal/
```

And you can use `git diff` to check the difference.

### our reproduced gr

`gr.py`

reproduce SEAL using title and substring as identifiers, `t5` only

the usage is similar to MVDR

run `bash gr_run_scripts/run.sh`

This will create an output diretory in `output/gr.DATA.train` by default, which is the `$EXP_DIR` defined in `gr_run_scripts/params.sh`

All GR related parameters are in `gr_run_scripts/params.sh`.

