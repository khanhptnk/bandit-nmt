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
