# Revisiting UID

## Install Dependencies

Create a conda environment with
```bash
$ conda env create -f environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install datasets transformers
```

## Get the Data

To get the data run
```bash
$ make get_data
```
Note that this does not include the Dundee corpus, for which the original authors have to be contacted!

## To estimate an N-gram model
First build the library in the kenlm submodule
```bash
$ cd kenlm
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j 4
```
then estimate the model from the wikitext 103 dataset
```bash
cat {data-dir}/wikitext-103/wiki.train.tokens | awk '!/=\s*/' | awk NF > /tmp/wiki.train.tokens.clean
bin/lmplz -o 5 --skip_symbols < /tmp/wiki.train.tokens.clean >wiki.arpa
```
