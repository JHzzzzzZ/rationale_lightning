embed_size=100;


python check_results.py --ckpt_path ./output/straight_beer_1-1-0.13-sgumbel-ns=1-noshare-2.0/best.ckpt \
       --embedding_path ./pretrained/glove.6B.${embed_size}d.txt \
       --embed_size $embed_size \
       --save_decoded_answer \
       --accelerator "gpu" \
       --devices "0," \
       --eval