# Revisiting UID

## Install Dependencies

Install requirements via pip
```bash
$ pip install -r requirements.txt
```

## Get the Data

To get the data run
```bash
$ cd src
bash pull_data.sh
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

You can then find the entire analysis pipeline in `src/revisiting-uid.ipynb`
