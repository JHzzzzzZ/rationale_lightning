devices="0,";

dataset=hotel;
aspect=0;
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
margin=0.3;
margin_weight=1;
num_samples=1;
ranking=margin;
gate=straight;

output_dir=./output/${gate}_${dataset}_${aspect}-rev-$margin-$ranking-$num_samples;
pretrained_dir=./output/baseline_${dataset}_${aspect}/best.ckpt;

python main.py --model rationale_rev --dataset $dataset \
       --embedding_path ./pretrained/glove.6B.${embed_size}d.txt \
       --embed_size $embed_size --fix_emb \
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
       --num_workers 0 \
       --shuffle_train \
       --devices $devices \
       --accelerator gpu \
       --max_epochs $epochs \
       --min_epochs $min_epochs \
       --precision 32 \
       --gradient_clip_val 5 \
       --output_dir $output_dir \
       --accumulate_grad_batches $(($total_batch_size/$batch_size)) \
       --monitor acc --monitor_mode max \
       --aspect $aspect \
       --gate $gate \
       --sparsity 1e-1 --coherence 10 \
       --selection 0.08 --lasso 0.02 \
       --margin $margin --margin_weight $margin_weight \
       --num_samples $num_samples --ranking $ranking \
       --unlimited_test_length \
       --val_is_test \
#       --val_check_interval 0.25 \
#       --decay 1e-2 \


#       --check_val_every_n_epoch 0.25 \
#       --reset_data \
#       --profiler simple \


       #       --fgm --fgm_epsilon 1 \
#       --pretrained_cls_ckpt_path $pretrained_dir --freeze_cls \

#       --reset_data
#       --balance \


#       --sentence_level
#--temperature 5 --min_temperature 1e-4 \










#       --sentence_level \
#       --temperature 0.5 --min_temperature 0.001 \



#       --debug
#./pretrained/glove.6B.300d.txt







