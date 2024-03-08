output_num=8
corpus=MSP-PODCAST1.11

model_type=wav2vec2-large-robust
num_classes=ALL 
seed=100
label_type=categorical
label_learning=soft-label 
epochs=50
batchsize=32 


for data_mode in primary; do
for label_rule in M P D; do
for partition_number in 1; do

corpus_type=${corpus}_${num_classes}_${data_mode}
model_path=model/${model_type}/${corpus_type}/${data_mode}/${label_rule}/partition${partition_number}/

for label_rule_test in M P D; do
    #Evaluation
    python -u test.py \
    --device            cuda \
    --model_type        $model_type \
    --corpus_type       $corpus_type \
    --seed              $seed \
    --batch_size        $batchsize \
    --hidden_dim        1024 \
    --num_layers        2 \
    --output_num        $output_num \
    --label_type        $label_type \
    --label_learning    $label_learning \
    --corpus            $corpus \
    --num_classes       $num_classes \
    --label_rule        $label_rule_test \
    --partition_number  $partition_number \
    --data_mode         $data_mode \
    --model_path        ${model_path}
done;
done;
done;
done