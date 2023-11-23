devices="0,";

dataset=hotel;
aspect=0;
loss_fn="ce";

optimizer=adamw;
lr=1e-5;
#weight_decay=1e-2;
weight_decay=0;

lr_scheduler=null; # exp, multi_step, null
gamma=0.97;
milestones=(15 50 80);

hidden_size=768;
dropout=0.1;
output_size=2;

min_epochs=30;
epochs=100;
total_batch_size=16;
batch_size=8;
max_len=256;
margin=0.02;
margin_weight=1;
num_samples=1;
ranking=null;
gate=gumbel;
sparsity=5e-4;
coherence=10;
selection=0.11;
lasso=0.0;
k=100;
threshold=0.2;

output_dir=./output/plm_kuma_beer_grow-$aspect;# -$margin-$ranking-$num_samples;
pretrained_dir=./output/baseline_beer-$aspect/best.ckpt;

for i in {1..1} ; do
    seed=$i
    output_dir=./output/plm_${gate}_${dataset}_${aspect}-${sparsity}-${coherence}-${selection}-${lasso}-${k}-${threshold}-$num_samples-${i};
    python main.py --model rationale --dataset $dataset --seed $seed \
       --model_path ./pretrained/bert-base-uncased \
       --optimizer $optimizer \
       --lr $lr \
       --weight_decay $weight_decay \
       --lr_scheduler $lr_scheduler \
       --gamma $gamma \
       --milestones "${milestones[@]}" \
       --hidden_size $hidden_size \
       --dropout $dropout \
       --output_size $output_size \
       --layer bert \
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
       --sparsity $sparsity --coherence $coherence \
       --selection $selection --lasso $lasso \
       --k k --threshold $threshold \
       --margin $margin --margin_weight $margin_weight \
       --num_samples $num_samples --ranking $ranking \
       --val_is_test \
#       --temperature 5 --min_temperature 1e-4 \
#       --decay 1e-3 \
#       --unlimited_test_length \

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



#       --sentence_level \



#       --debug
#./pretrained/glove.6B.300d.txt

done




