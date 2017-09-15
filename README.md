# bandit-nmt

Code will be ready to use soon.

Set home directory:

~~~~
> export BANDIT_HOME=$PWD
> echo $BANDIT_HOME
~~~~

### Data extraction

Download pre-processing scripts:

~~~~
> cd $BANDIT_HOME/data/scripts
> bash download_scripts.sh
~~~~

For German-English:
~~~~
> cd $BANDIT_HOME/data/en-de
> bash extract_data_de_en.sh
~~~~

Data should be ready in `$BAND_HOME/data/en-de/prep`

TODO: Chinese-English needs segmentation
