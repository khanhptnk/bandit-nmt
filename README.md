# bandit-nmt

Code will be ready to use soon.

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

### Reinforcement training from a pretrained model

~~~~
> cd $BANDIT_HOME
> python train.py -data $PATH_TO_YOUR_DATA -load_from $PATH_TO_YOUR_MODEL -save_dir $YOUR_SAVE_DIR -start_reinforce -1 -end_epoch 100 -critic_pretrain_epochs 1
~~~~
