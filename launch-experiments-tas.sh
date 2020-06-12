#!/bin/bash

if [ ! -z $1 ]
then
    root_data="./data"
    dir_tas="${root_data}/twitter_airline_sentiment"
    zip_tas="./${root_data}/twitter-airline-sentiment.zip"

    rm -rf ${dir_tas}
    mkdir -p ${dir_tas}
    unzip ${zip_tas} -d ${dir_tas}
fi

datasets=('twitter_airline_sentiment')
model_types=('bert' 'distilbert')
output_dir='./saved_models'
batch_size=128
max_length=64
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