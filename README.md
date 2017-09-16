# bandit-nmt

**NOTE**: as of Sep 16, the code got 3x slower when I upgraded to PyTorch 2.0. This is a known issue and [PyTorch is fixing it](https://github.com/pytorch/pytorch/issues/2518#issuecomment-327835296). 

Set home directory:

~~~~
> export BANDIT_HOME=$PWD
> export DATA=$BANDIT_HOME/data
> export SCRIPT=$BANDIT_HOME/scripts
~~~~

### Data extraction

Download pre-processing scripts:

~~~~
> cd $DATA/scripts
> bash download_scripts.sh
~~~~

For German-English:
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

### Pretraining actor and critic

~~~~
> cd $SCRIPT
> bash pretrain.sh en-de $YOUR_LOG_DIR
~~~~

See `scripts/pretrain.sh` for more details.

To simply supervisedly train a model for 10 epochs

~~~~
> cd $BANDIT_HOME
> python train.py -data $YOUR_DATA -save_dir $YOUR_SAVE_DIR -end_epoch 10
~~~~

### Reinforcement training from a pretrained model

~~~~
> cd $BANDIT_HOME
> python train.py -data $YOUR_DATA -load_from $YOUR_MODEL -save_dir $YOUR_SAVE_DIR -start_reinforce -1 -end_epoch 100 -critic_pretrain_epochs 1
~~~~
