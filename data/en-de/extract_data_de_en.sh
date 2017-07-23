#!/usr/bin/env bash

echo "Download data..."
echo ""

wget -O train_2015.tgz https://wit3.fbk.eu/archive/2015-01//texts/en/de/en-de.tgz
wget -O train_2014.tgz https://wit3.fbk.eu/archive/2014-01//texts/en/de/en-de.tgz

tar -xvzf train_2015.tgz
mkdir -p orig

cp en-de/*.xml orig/
cp en-de/train.tags.en-de.en orig/train_2015.tags.en-de.en
cp en-de/train.tags.en-de.de orig/train_2015.tags.en-de.de
rm -rf en-de/

tar -xvzf train_2014.tgz 
cp en-de/train.tags.en-de.en orig/train_2014.tags.en-de.en
cp en-de/train.tags.en-de.de orig/train_2014.tags.en-de.de

rm -rf en-de/

echo "Preprocessing..."
echo ""

SCRIPTS=../scripts
TOKENIZER=$SCRIPTS/tokenizer.perl
LC=$SCRIPTS/lowercase.perl
CLEAN=$SCRIPTS/clean-corpus-n.perl

src=de
tgt=en
lang=en-de

prep=prep
tmp=prep/tmp
orig=orig

mkdir -p $tmp $prep

#cd $orig
#tar zxvf $GZ
#cd -

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
        perl $LC < $tmp/train_$year.$lang.clean.$l > $tmp/train_$year.$lang.$l
    done
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
    awk '{if (NR%25 == 0)  print $0; }' $tmp/train_2015.$lang.$l > $prep/valid.$lang.$l
    awk '{if (NR%25 != 0)  print $0; }' $tmp/train_2015.$lang.$l > $prep/train_2015.$lang.$l
    cat $prep/train_2014.$lang.$l $prep/train_2015.$lang.$l > $prep/train.$lang.$l

    echo "Train 2014: $(wc -l $prep/train_2014.$lang.$l)"
    echo "Train 2015: $(wc -l $prep/train_2015.$lang.$l)"
    echo "Valid     : $(wc -l $prep/valid.$lang.$l)"

    cat $tmp/IWSLT15.TED.dev2010.$lang.$l \
        $tmp/IWSLT15.TEDX.dev2012.$lang.$l \
        $tmp/IWSLT15.TED.tst2010.$lang.$l \
        $tmp/IWSLT15.TED.tst2011.$lang.$l \
        $tmp/IWSLT15.TED.tst2012.$lang.$l \
	$tmp/IWSLT15.TED.tst2013.$lang.$l \
	$tmp/IWSLT15.TEDX.tst2013.$lang.$l \
        > $prep/test.$lang.$l

    echo "Test      : $(wc -l $prep/test.$lang.$l)"
    echo ""
done

rm -rf $tmp

echo "DONE!!!"
