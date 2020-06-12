#!/bin/bash

threshold=512

if [ ! -z $1 ]
then
    root_data="./data/20newsgroups"
    rm -rf ${root_data}
    mkdir -p ${root_data}
    wget -P ${root_data} "http://qwone.com/~jason/20Newsgroups/vocabulary.txt"
    wget -P ${root_data} "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    tmp_dir="/tmp/20newsgroups"
    mkdir -p ${tmp_dir}
    compressed_file="${root_data}/20news-bydate.tar.gz"
    tar -xvf ${compressed_file} -C ${tmp_dir}
    rm -rf ${compressed_file}
    sections=("20news-bydate-train" "20news-bydate-test")
    for section in "${sections[@]}"
    do
        dir_section="${tmp_dir}/$section"
        groups=($(ls ${dir_section}/))
        for group in "${groups[@]}"
        do
            dir_group=${dir_section}/$group
            samples=($(ls ${dir_group}/))
            dir_unicode_group=${root_data}/${section}/${group}
            mkdir -p ${dir_unicode_group}
            for sample in "${samples[@]}"
            do
                input="${dir_group}/$sample"
                output="${dir_unicode_group}/$sample.txt"
                iconv -f iso-8859-1 -t utf-8 < $input > $output
            done
        done
        rm -rf ${dir_section}
    done
    python3 preprocess_20newsgroups.py ${root_data} --threshold ${threshold}
fi

datasets=('6_simplified' '20_simplified')
model_types=('bert' 'distilbert')
output_dir='./saved_models'
batch_size=16
max_length=512
learning_rate=5e-5
adam_epsilon=1e-8
weight_decay=0.1
num_train_epochs=10
num_warmup_steps=0
max_grad_norm=1

for dataset in "${datasets[@]}"
do
    for model_type in "${model_types[@]}"
    do
        echo "${dataset} - ${model_type}"
        python3 training_plmc.py ${dataset} ${model_type} --output_dir ${output_dir} --batch_size ${batch_size} \
        --max_length ${max_length} --learning_rate ${learning_rate} --adam_epsilon ${adam_epsilon} --num_train_epochs ${num_train_epochs} \
        --num_warmup_steps ${num_warmup_steps} --max_grad_norm ${max_grad_norm} --weight_decay ${weight_decay} --lowercase
    done
done