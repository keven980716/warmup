ARCH=transformer_iwslt_de_en
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined

LR=0.1
GPU=0
WU=5000

for SEED in 1234 2345 3456 4567 5678
do

echo "seed=$SEED"
echo "warm up updates=$WU"
OUTPUT_PATH=/data2/checkpoints_sgd/IWSLT/post-norm-sgd-nowu-$LR-seed$SEED
RESULT_PATH=/data2/results_sgd/IWSLT/post-norm-sgd-nowu-$LR-seed$SEED

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH


#CUDA_VISIBLE_DEVICES=$GPU python3 train.py $DATA_PATH  \
#    --seed $SEED   \
#    -a $ARCH  --share-all-embeddings \
#    --optimizer sgd --momentum 0.9  --lr $LR --fp16 \
#    -s de -t en \
#    --clip-norm 0.0 \
 #   --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
 #   --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
 #   --criterion label_smoothed_cross_entropy \
#    --max-update 50000 --keep-last-epochs 20 \
#    --warmup-updates $WU --warmup-init-lr 1e-7  \
#    --save-dir $OUTPUT_PATH \
#    --no-progress-bar --log-interval 100 \
#    --ddp-backend=no_c10d 2>&1 | tee -a $OUTPUT_PATH/train_log-gp.txt 
    
    
python3 scripts/average_checkpoints.py --inputs $OUTPUT_PATH \
  --num-epoch-checkpoints 10 --output $OUTPUT_PATH/avg_10.pt
   
python3 generate.py $DATA_PATH \
  --path $OUTPUT_PATH/avg_10.pt \
  --log-format simple \
  --batch-size 128 --beam 5 --remove-bpe --lenpen 1.0 \
  > $RESULT_PATH/avg_10.txt
  
python3 generate.py $DATA_PATH \
  --path $OUTPUT_PATH/checkpoint_best.pt \
  --log-format simple \
  --batch-size 128 --beam 5 --remove-bpe --lenpen 1.0 \
  > $RESULT_PATH/checkpoint_best.txt
done 
