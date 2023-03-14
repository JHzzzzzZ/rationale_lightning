devices="0,";

dataset=hotel;
aspect=1;
loss_fn="ce";
#if [ $dataset == beer ]
#then
#  loss_fn="mse"
#else
#  loss_fn="bce"
#fi

embed_size=100;
hidden_size=100;
dropout=0;
output_size=2;

optimizer=adam;
lr=1e-3;
#weight_decay=2e-6;
weight_decay=0;

min_epochs=30;
epochs=100;
total_batch_size=256;
batch_size=256;
max_len=512;

output_dir=./output/baseline_${dataset}_${aspect};

python main.py --model baseline --dataset $dataset \
       --embedding_path ./pretrained/glove.6B.${embed_size}d.txt \
       --embed_size $embed_size \
       --fix_emb \
       --optimizer $optimizer \
       --lr $lr \
       --weight_decay $weight_decay \
       --lr_scheduler exp \
       --gamma 0.97 \
       --hidden_size $hidden_size \
       --dropout $dropout \
       --output_size $output_size \
       --layer lstm \
       --loss_fn $loss_fn \
       --data_path ./data/$dataset \
       --max_len $max_len \
       --batch_size $batch_size \
       --num_workers 10 \
       --shuffle_train \
       --devices $devices \
       --accelerator gpu \
       --max_epochs $epochs \
       --min_epochs $min_epochs \
       --precision 32 \
       --gradient_clip_val 5 \
       --output_dir $output_dir \
       --aspect $aspect \
#       --fgm --fgm_epsilon 1
#       --gate gumbel \
#       --decay 0.01 \
#       --sparsity 0.1 --coherence 0.01 \
#       --selection 0.13 --lasso 0.02 \
#       --temperature 0.5 --min_temperature 0.05 \

#       --pretrained_cls_ckpt_path output/model.ckpt \
#       --debug



#       --flat_doc --concat_query \
#       --embedding_path ../pretrained/glove.6B.300d.txt \
#       --profiler simple \





