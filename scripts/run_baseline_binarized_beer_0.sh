devices="0,";

dataset=beer;
aspect=0;
loss_fn="ce";

embed_size=100;
hidden_size=100;
dropout=0.2;
output_size=2;

optimizer=adam; # adam, adamw, sgd
lr=1e-4;
#weight_decay=2e-6;
weight_decay=0;

lr_scheduler=null; # exp, multi_step, null
gamma=0.97;
milestones=(15 50 80);

min_epochs=300;
epochs=300;
total_batch_size=256;
batch_size=256;
max_len=256;


pretrained_dir=./output/baseline_${dataset}_${aspect}/best.ckpt;


for i in {1..1} ; do
    seed=$i
    output_dir=./output/${baseline}_${dataset}_${aspect};
    python main.py --model baseline --dataset $dataset --seed $seed \
       --embedding_path ./pretrained/glove.6B.${embed_size}d.txt \
       --embed_size $embed_size --fix_emb \
       --optimizer $optimizer \
       --lr $lr \
       --weight_decay $weight_decay \
       --lr_scheduler $lr_scheduler \
       --gamma $gamma \
       --milestones "${milestones[@]}" \
       --hidden_size $hidden_size \
       --dropout $dropout \
       --output_size $output_size \
       --layer gru \
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
       --balance \
       --unlimited_test_length \
       --val_is_test \
#       --straight_with_gumbel \
#       --share
#       --auxiliary_full_pred \
#       --num_samples $num_samples \
#       --ranking $ranking \
#       --margin $margin --margin_weight $margin_weight \
#       --share \





#       --num_samples $num_samples --ranking $ranking \
#       --margin $margin --margin_weight $margin_weight \

#       --val_check_interval 0.25 \
#       --decay 1e-2 \
#       --k $k --threshold $threshold \


#       --check_val_every_n_epoch 0.25 \
#       --reset_data \
#       --profiler simple \

#

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

done







