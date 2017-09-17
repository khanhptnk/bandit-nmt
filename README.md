# bandit-nmt

This is code repo for our EMNLP 2017 paper ["Reinforcement Learning for Bandit Neural Machine Translation with Simulated Human Feedback"](https://arxiv.org/pdf/1707.07402.pdf), which implements the [A2C algorithm](https://arxiv.org/pdf/1602.01783.pdf) on top of a [neural encoder-decoder model](https://arxiv.org/pdf/1508.04025.pdf) and benchmarks the combination under simulated noisy rewards. 

**NOTE**: as of Sep 16, the code got 3x slower when I upgraded to PyTorch 2.0. This is a known issue and [PyTorch is fixing it](https://github.com/pytorch/pytorch/issues/2518#issuecomment-327835296). 

Set home directory:

~~~~
> export BANDIT_HOME=$PWD
> export DATA=$BANDIT_HOME/data
> export SCRIPT=$BANDIT_HOME/scripts
~~~~

### Data extraction

Download pre-processing scripts
~~~~
> cd $DATA/scripts
> bash download_scripts.sh
~~~~

For German-English
~~~~
> cd $DATA/en-de
> bash extract_data_de_en.sh
~~~~

Data should be ready in `$BAND_HOME/data/en-de/prep`

TODO: Chinese-English needs segmentation


### Data pre-processing

~~~~
> cd $SCRIPT
> bash make_data.sh de en
~~~~

### Pretraining

Pretrain both actor and critic
~~~~
> cd $SCRIPT
> bash pretrain.sh en-de $YOUR_LOG_DIR
~~~~

See `scripts/pretrain.sh` for more details.

Pretrain actor only
~~~~
> cd $BANDIT_HOME
> python train.py -data $YOUR_DATA -save_dir $YOUR_SAVE_DIR -end_epoch 10
~~~~

### Reinforcement training 

~~~~
> cd $BANDIT_HOME
~~~~

From scratch
~~~~
> python train.py -data $YOUR_DATA -load_from $YOUR_MODEL -save_dir $YOUR_SAVE_DIR -start_reinforce 10 -end_epoch 100 -critic_pretrain_epochs 5
~~~~

From a pretrained model
~~~~
> python train.py -data $YOUR_DATA -load_from $YOUR_MODEL -save_dir $YOUR_SAVE_DIR -start_reinforce -1 -end_epoch 100 -critic_pretrain_epochs 5
~~~~


