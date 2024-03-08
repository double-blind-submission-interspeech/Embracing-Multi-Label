corpus=MSP-IMPROV

model_type=wav2vec2-large-robust
seed=100
label_type=categorical
label_learning=soft-label 
epochs=50
batchsize=32


for label_rule in M P D; do
for partition_number in 1 2 3 4 5 6; do
for data_mode in primary; do
if [[ ${data_mode} == "primary" ]]; then
    output_num=4
    num_classes=four 
elif [[ ${data_mode} == "secondary" ]]; then
    output_num=10
    num_classes=ALL 
fi
corpus_type=${corpus}_${num_classes}_${data_mode}
# Training
python -u train.py \
--device            cuda \
--lr                1e-4 \
--model_type        $model_type \
--corpus_type       $corpus_type \
--seed              $seed \
--epochs            $epochs \
--batch_size        $batchsize \
--hidden_dim        1024 \
--num_layers        2 \
--output_num        $output_num \
--label_type        $label_type \
--label_learning    $label_learning \
--corpus            $corpus \
--num_classes       $num_classes \
--label_rule        $label_rule \
--partition_number  $partition_number \
--data_mode         $data_mode \
--model_path        model/${model_type}/${corpus_type}/${data_mode}/${label_rule}/partition${partition_number}/

done;
done;
done