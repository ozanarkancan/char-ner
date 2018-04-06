## CharNER
For more details, please refer to the paper: [CharNER: Character-Level Named Entity Recognition](http://www.aclweb.org/anthology/C/C16/C16-1087.pdf)

## Usage

Access to full list of options by typing:
```
python exper.py --help
```

Example command:
```
python src/exper.py --activation bi-lstm --n_hidden 128 128 --drates .2 .5 .8 --lang cze
```
This command builds 2 Bidirectional LSTMs stacked of top of each other.
Each forward and backward LSTM has 128 units.
--drates (dropout rates) flag signals to use dropout.
In this example, .2 dropout is applied to inputs (drops characters) and .5 & .8 dropouts are applied to the outputs of Bidirectional LSTMs. 
--lang flag dictates which folder to use under data/ directory

## Data Format
Each folder under data/ directory is composed of 3 files.
train.bio, testa.bio, testb.bio are for training, development and test sets respectively.
Each file contains word, tag pairs seperated with a tab.
For examples, check out directories under data/. 

## Install Dependencies
```
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install -r requirements.txt
```

## Authors
* Onur Kuru
* Ozan Arkan Can
