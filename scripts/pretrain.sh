lang=$1
DATA_DIR=$DATA/$lang
SAVE_DIR=$2

if [ ! -d "$DATA_DIR" ]; then
    echo "Can't find data dir $DATA_DIR!"
    echo "**First argument is either en-de or en-zh!"
    exit 1
fi

if [ -d "$SAVE_DIR" ]; then 
    echo "$SAVE_DIR existed!"
    echo "**Please choose another saving directory!"
    exit 1
fi

mkdir -p $SAVE_DIR

python -u ../train.py -data $DATA_DIR/processed_all-train.pt \
                      -start_reinforce 11 \
                      -critic_pretrain_epochs 5 \
                      -end_epoch 15 \
                      -save_dir $SAVE_DIR
