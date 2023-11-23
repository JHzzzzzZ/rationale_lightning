devices="4,5,6,7";

dataset=fever;
loss_fn="ce";

embed_size=100;
hidden_size=100;
dropout=0;
output_size=2;

optimizer=adam; # adam, adamw, sgd
lr=1e-3;
#weight_decay=2e-6;
weight_decay=0;

lr_scheduler=null; # exp, multi_step, null
gamma=0.97;
milestones=(15 50 80);

min_epochs=30;
epochs=100;
total_batch_size=256;
batch_size=256;
max_len=512;
#margin=0.02;
#margin_weight=1;
num_samples=4;
#ranking=margin;
gate=gumbel;
sparsity=5e-2;
coherence=0;
selection=0.40;
lasso=0.0;


pretrained_dir=./output/baseline_${dataset}_${aspect}/best.ckpt;


for i in {1..1} ; do
    seed=$i
    output_dir=./output/${gate}_${dataset}-${sparsity}-${coherence}-${selection}-${lasso}-${num_samples}-${i};
    python main.py --model rationale --dataset $dataset --seed $seed \
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
       --gate $gate \
       --sparsity $sparsity --coherence $coherence \
       --selection $selection --lasso $lasso \
       --decay 1e-2 \
       --num_samples $num_samples \
       --sentence_level \
#       --unlimited_test_length \

#       --val_check_interval 0.25 \



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







