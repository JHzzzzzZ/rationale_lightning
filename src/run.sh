devices="1,";

dataset=beer;
loss_fn="mse";
#if [ $dataset == beer ]
#then
#  loss_fn="mse"
#else
#  loss_fn="bce"
#fi

optimizer=adam;
lr=1e-4;
weight_decay=2e-6;
#weight_decay=0;

epochs=100;
batch_size=128;
max_len=512;
margin=0.02;
margin_weight=1;
num_samples=9;
ranking=margin;
aspect=0;

output_dir=./output/kuma_beer_rev_grow-$aspect-$margin-$ranking-$num_samples--;
pretrained_dir=./output/baseline_beer_$aspect/best.ckpt;

python main.py --model rationale_rev --dataset $dataset \
       --embedding_path ./pretrained/glove.6B.300d.txt \
       --embed_size 300 --fix_emb \
       --optimizer $optimizer \
       --lr $lr \
       --weight_decay $weight_decay \
       --lr_scheduler exp \
       --gamma 0.97 \
       --hidden_size 400 \
       --dropout 0.2 \
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
       --precision 32 \
       --gradient_clip_val 5 \
       --output_dir $output_dir \
       --accumulate_grad_batches 2 \
       --monitor loss \
       --aspect $aspect \
       --gate kuma \
       --sparsity 1e-3 --coherence 1e-3 \
       --selection 0.13 --lasso 0.02 \
       --decay 1e-3 \
       --margin $margin --margin_weight $margin_weight \
       --num_samples $num_samples --ranking $ranking \
#       --pretrained_cls_ckpt_path $pretrained_dir \
#       --freeze_cls \

#       --reset_data
#       --balance \


#       --sentence_level
#--temperature 5 --min_temperature 1e-4 \










#       --sentence_level \
#       --temperature 0.5 --min_temperature 0.001 \



#       --debug
#./pretrained/glove.6B.300d.txt

#       --profiler simple \





