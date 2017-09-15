src=$1
tgt=$2
lang=${2}-${1}

export DATA_HOME=$DATA/$lang
export DATA_PREP=$DATA_HOME/prep

python ../preprocess.py \
  -train_src $DATA_PREP/train.$lang.$src \
  -train_tgt $DATA_PREP/train.$lang.$tgt \
  -train_xe_src $DATA_PREP/train_2015.$lang.$src \
  -train_xe_tgt $DATA_PREP/train_2015.$lang.$tgt \
  -train_pg_src $DATA_PREP/train_2014.$lang.$src \
  -train_pg_tgt $DATA_PREP/train_2014.$lang.$tgt \
  -valid_src $DATA_PREP/valid.$lang.$src \
  -valid_tgt $DATA_PREP/valid.$lang.$tgt \
  -test_src $DATA_PREP/test.$lang.$src \
  -test_tgt $DATA_PREP/test.$lang.$tgt \
  -save_data $DATA_HOME/processed_all
