devices="0,";

dataset=beer;
if [ $dataset == beer ]
then
  loss_fn="mse"
else
  loss_fn="bce"
fi

output_dir=./output/kuma_beer_rev_grow-set_margin=$margin;
pretrained_dir=./output/baseline_beer/best.ckpt

optimizer=adam;
lr=1e-4;
weight_decay=2e-6;
#weight_decay=0;

epochs=100;
batch_size=128;
max_len=512;
margin=0.09;
margin_weight=1;

python main.py --model rationale_rev --dataset $dataset \
       --embedding_path ../pretrained/glove.6B.300d.txt \
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
       --gate kuma \
       --decay 1e-3 \
       --sparsity 1e-3 --coherence 0.001 \
       --selection 0.13 --lasso 0.02 \
       --monitor loss \
       --aspect 0 \
       --margin $margin --margin_weight $margin_weight \
#       --num_samples 1
#       --sentence_level
#--temperature 5 --min_temperature 1e-4 \

#       --pretrained_cls_ckpt_path $pretrained_dir








#       --sentence_level \
#       --temperature 0.5 --min_temperature 0.001 \






#       --aspect 0 \


#       --debug
#./pretrained/glove.6B.300d.txt

#       --flat_doc --concat_query \
#       --profiler simple \





