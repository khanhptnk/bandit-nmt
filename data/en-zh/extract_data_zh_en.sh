#!/usr/bin/env bash

wget -O train_2015.tgz https://wit3.fbk.eu/archive/2015-01//texts/en/zh/en-zh.tgz
wget -O train_2014.tgz https://wit3.fbk.eu/archive/2014-01//texts/en/zh/en-zh.tgz
wget -O eval_2014.tgz https://wit3.fbk.eu/archive/2014-01-test//texts/en/zh/en-zh.tgz

if [ ! -d "stanford-segmenter-*"]; then
    wget https://nlp.stanford.edu/software/stanford-segmenter-2017-06-09.zip
    unzip stanford-segmenter-2017-06-09.zip
fi

mkdir -p orig
tar -xvzf train_2015.tgz

cp en-zh/*.xml orig/
cp en-zh/train.tags.en-zh.en orig/train_2015.tags.en-zh.en
cp en-zh/train.tags.en-zh.zh orig/train_2015.tags.en-zh.zh
rm -rf en-zh/

tar -xvzf train_2014.tgz
cp en-zh/*.xml orig/
cp en-zh/train.tags.en-zh.en orig/train_2014.tags.en-zh.en
cp en-zh/train.tags.en-zh.zh orig/train_2014.tags.en-zh.zh
rm -rf en-zh/

tar -xvzf eval_2014.tgz
cp en-zh/*.xml orig/
rm -rf en-zh/

SCRIPTS=../scripts
TOKENIZER=$SCRIPTS/tokenizer.perl
LC=$SCRIPTS/lowercase.perl
CLEAN=$SCRIPTS/clean-corpus-n.perl
ZH_TOKENIZER=./stanford-segmenter-*/segment.sh

src=zh
tgt=en
lang=en-zh

prep=prep
tmp=prep/tmp
orig=orig

mkdir -p $tmp $prep

echo "pre-processing train data..."
for year in 2014 2015; do
    for l in $src $tgt; do
        f=train_$year.tags.$lang.$l
        tok=train_$year.$lang.tok.$l

        cat $orig/$f | \
        grep -v '<url>' | \
        grep -v '<talkid>' | \
        grep -v '<keywords>' | \
        sed -e 's/<title>//g' | \
        sed -e 's/<\/title>//g' | \
        sed -e 's/<description>//g' | \
        sed -e 's/<\/description>//g' | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
        
        echo ""
    done
    perl $CLEAN $tmp/train_$year.$lang.tok $src $tgt $tmp/train_$year.$lang.clean 1 50
    for l in $src $tgt; do
        if [ $l == $tgt ]; then
            outfile=train_$year.$lang.$l
        else    
            outfile=train_$year.tmp.$lang.$l
        fi
        perl $LC < $tmp/train_$year.$lang.clean.$l > $tmp/$outfile
    done
    $ZH_TOKENIZER ctb $tmp/train_$year.tmp.$lang.$src UTF-8 0 > $tmp/train_$year.$lang.$src
    wc -l $tmp/train_$year.$lang.$src
    wc -l $tmp/train_$year.$lang.$tgt
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/IWSLT*.TED*.$l.xml`; do
        fname=${o##*/}
        f=$tmp/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
                sed -e 's/<seg id="[0-9]*">\s*//g' | \
                sed -e 's/\s*<\/seg>\s*//g' | \
                sed -e "s/\’/\'/g" | \
                perl $TOKENIZER -threads 8 -l $l | \
                perl $LC > $f
    echo ""
    done
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    cat $tmp/train_2014.$lang.$l > $prep/train_2014.$lang.$l
    awk '{if (NR%25 == 0)  print $0; }' $tmp/train_2015.$lang.$l > $tmp/valid.tmp.$lang.$l
    awk '{if (NR%25 != 0)  print $0; }' $tmp/train_2015.$lang.$l > $prep/train_2015.$lang.$l

    cat $tmp/train_2014.$lang.$l $tmp/train_2015.$lang.$l > $prep/train.$lang.$l

    cat $tmp/IWSLT15.TED.dev2010.$lang.$l \
        $tmp/IWSLT15.TED.tst2010.$lang.$l \
        $tmp/IWSLT15.TED.tst2011.$lang.$l \
        $tmp/IWSLT15.TED.tst2012.$lang.$l \
        $tmp/IWSLT14.TED.tst2013.$lang.$l \
        $tmp/IWSLT14.TED.tst2014.$lang.$l \
        > $tmp/test.tmp.$lang.$l

done

for set in valid test; do
    perl $CLEAN $tmp/$set.tmp.$lang $src $tgt $tmp/$set.tmp.$lang.clean 1 50
    $ZH_TOKENIZER ctb $tmp/$set.tmp.$lang.$src UTF-8 0 > $prep/$set.$lang.$src

    cp $tmp/$set.tmp.$lang.$tgt $prep/$set.$lang.$tgt

done

for l in $src $tgt; do
    echo "Train 2014: $(wc -l $prep/train_2014.$lang.$l)"
    echo "Train 2015: $(wc -l $prep/train_2015.$lang.$l)"
    echo "Valid     : $(wc -l $prep/valid.$lang.$l)"
    echo "Test      : $(wc -l $prep/test.$lang.$l)"
    echo ""
done


