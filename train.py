# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
from tqdm import tqdm

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WavLMModel

# Self-Written Modules
sys.path.append(os.getcwd())
import utils
import net
# from net import ser, chunk



def main(args):
    #utils.set_deterministic(args.seed)
    utils.print_config_description(args.conf_path)

    config_dict = utils.load_env(args.conf_path)
    assert config_dict.get("config_root", None) != None, "No config_root in config/conf.json"
    # assert config_dict.get(args.corpus_type, None) != None, "Change config/conf.json"
    config_path = os.path.join(config_dict["config_root"], config_dict[args.corpus_type])
    utils.print_config_description(config_path)

    # Make model directory
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)


    # Initialize dataset
    DataManager=utils.DataManager(config_path)
    lab_type = args.label_type
    # print(lab_type)
    if args.label_type == "dimensional":
        assert args.output_num == 3

    if args.label_type == "categorical":
        emo_num = DataManager.get_categorical_emo_num()
        # print(emo_num)
        assert args.output_num == emo_num

    audio_path, label_path = utils.load_audio_and_label_file_paths(args)
    snum=10000000000000000
    train_wav_path = DataManager.get_wav_path(split_type="train",wav_loc=audio_path, lbl_loc=label_path)[:snum]
    train_utts = DataManager.get_utt_list("train",lbl_loc=label_path)[:snum]
    train_labs = DataManager.get_msp_labels(train_utts, lab_type=lab_type,lbl_loc=label_path)
    train_wavs = utils.WavExtractor(train_wav_path).extract()

    dev_wav_path = DataManager.get_wav_path(split_type="dev",wav_loc=audio_path,lbl_loc=label_path)[:snum]
    dev_utts = DataManager.get_utt_list("dev",lbl_loc=label_path)[:snum]
    dev_labs = DataManager.get_msp_labels(dev_utts, lab_type=lab_type,lbl_loc=label_path)
    dev_wavs = utils.WavExtractor(dev_wav_path).extract()
    ###################################################################################################

    # Designed loss
    # print(train_labs)
    train_ori = np.asarray(train_labs)
    # print(train_ori,train_ori.shape)
    MR_count = 0
    PR_count = 0
    AR_count = 0

    for idx, each in enumerate(train_ori):
        cur_train = train_ori[idx,:] # [0.01     0.532222 0.01     0.323333 0.114444 0.01    ]
        max_index = np.where(np.max(cur_train)==cur_train)[0]
        # print(cur_train,max_index,np.max(cur_train))
        if len(max_index) == 1 and np.max(cur_train)>0.5:
                MR_count+=1
        elif len(max_index) == 1:
                PR_count+=1
        else:
            AR_count+=1
    
    if MR_count ==0:
        MR_count=1
    if PR_count ==0:
        PR_count=1
    if AR_count ==0:
        AR_count=1                
    
    rule_per_cls = torch.Tensor(np.array([MR_count,PR_count,AR_count]))

    beta_loss = (train_labs.shape[0]-1)/train_labs.shape[0]
    no_of_classes_loss = 3
    effective_num_loss = 1.0 - torch.pow(beta_loss, rule_per_cls)
    weights_loss = (1.0 - beta_loss) / effective_num_loss
    rule_balanced_weights = weights_loss / torch.sum(weights_loss) * no_of_classes_loss

    print("rule_per_cls",rule_per_cls)
    print("rule_balanced_weights",rule_balanced_weights)


    # Class balanced weights
    k_thresold = 1/train_labs.shape[1]
    train_labs_PT = torch.Tensor(np.asarray(train_labs))
    train_labs_binary_PT = torch.where(train_labs_PT>k_thresold,1.0,0.0)

    samples_per_cls = torch.sum(train_labs_binary_PT,dim=0)
    samples_per_cls = [1.0 if x == 0.0 else x for x in samples_per_cls]
    samples_per_cls = torch.Tensor(np.array(samples_per_cls))

    beta = (train_labs.shape[0]-1)/train_labs.shape[0]
    no_of_classes = train_labs.shape[1]
    effective_num = 1.0 - torch.pow(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    class_balanced_weights = weights / torch.sum(weights) * no_of_classes
    print("samples_per_cls",samples_per_cls)
    print("self.class_balanced_weights",class_balanced_weights)

    train_set = utils.WavSet(train_wavs, train_labs, train_utts, 
        print_dur=True, lab_type=lab_type,
        label_config = DataManager.get_label_config(lab_type)
    )
    dev_set = utils.WavSet(dev_wavs, dev_labs, dev_utts, 
        print_dur=True, lab_type=lab_type,
        wav_mean = train_set.wav_mean, wav_std = train_set.wav_std,
        label_config = DataManager.get_label_config(lab_type)
    )
    train_set.save_norm_stat(model_path+"/train_norm_stat.pkl")
    
    total_dataloader={
        "train": DataLoader(train_set, batch_size=args.batch_size, collate_fn=utils.collate_fn_padd, shuffle=True, drop_last=True),
        "dev": DataLoader(dev_set, batch_size=args.batch_size, collate_fn=utils.collate_fn_padd, shuffle=False, drop_last=True)
    }

    # Initialize model
    modelWrapper = net.ModelWrapper(args) # Change this to use custom model
    modelWrapper.init_model()
    modelWrapper.init_optimizer()
    
    # Initialize loss function
    lm = utils.LogManager()
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list(["train_aro", "train_dom", "train_val",
            "dev_aro", "dev_dom", "dev_val"])
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc"])

    epochs=args.epochs
    scaler = GradScaler()
    min_epoch = 0
    min_loss = 99999999999
    temp_dev = 99999999999
    losses_train, losses_dev = [], []
    for epoch in range(epochs):
        print("Epoch:",epoch)
        lm.init_stat()
        modelWrapper.set_train()
        for xy_pair in tqdm(total_dataloader["train"]):
            x = xy_pair[0]
            y = xy_pair[1]
            mask = xy_pair[2]

            x=x.cuda(non_blocking=True).float()
            y=y.cuda(non_blocking=True).float()
            mask=mask.cuda(non_blocking=True).float()

            
            with autocast():
                ## Feed-forward
                pred = modelWrapper.feed_forward(x, attention_mask=mask)
                
                ## Calculate loss
                total_loss = 0.0
                if args.label_type == "dimensional":
                    ccc = utils.CCC_loss(pred, y)
                    loss = 1.0-ccc
                    total_loss += loss[0] + loss[1] + loss[2]
                elif args.label_type == "categorical":
                    if args.label_learning == "hard-label":
                        loss = utils.CE_category(pred, y)
                    elif args.label_learning == "soft-label":
                        # loss = utils.SCE_category(pred, y)
                        loss = utils.class_balanced_softmax_cross_entropy_with_softtarget(pred, y, class_balanced_weights.cuda(non_blocking=True).float())
                        # loss = utils.all_rule_loss(pred, y, class_balanced_weights.cuda(non_blocking=True).float(),rule_balanced_weights.cuda(non_blocking=True).float())
                        # loss = utils.mr_all_rule_loss(pred, y, class_balanced_weights.cuda(non_blocking=True).float(),rule_balanced_weights.cuda(non_blocking=True).float())
                    elif args.label_learning == "multi-label":
                        loss = utils.BCE_category(pred, y)
                    elif args.label_learning == "distribution-label":
                        loss = utils.KLD_category(pred, y)                    
                    #loss = utils.CE_category(pred, y)
                    total_loss += loss
                    # acc = utils.calc_acc(pred, y)
                    acc = utils.macro_f1(pred, y)
                    

            ## Backpropagation
            modelWrapper.backprop(total_loss)

            # Logging
            if args.label_type == "dimensional":
                lm.add_torch_stat("train_aro", ccc[0])
                lm.add_torch_stat("train_dom", ccc[1])
                lm.add_torch_stat("train_val", ccc[2])
            elif args.label_type == "categorical":
                lm.add_torch_stat("train_loss", loss)
                lm.add_torch_stat("train_acc", acc)

        modelWrapper.set_eval()

        with torch.no_grad():
            total_pred = [] 
            total_y = []
            for xy_pair in tqdm(total_dataloader["dev"]):
                x = xy_pair[0]
                y = xy_pair[1]
                mask = xy_pair[2]

                x=x.cuda(non_blocking=True).float()
                y=y.cuda(non_blocking=True).float()
                mask=mask.cuda(non_blocking=True).float()

                pred = modelWrapper.feed_forward(x, attention_mask=mask, eval=True)
                total_pred.append(pred)
                total_y.append(y)

            total_pred = torch.cat(total_pred, 0)
            total_y = torch.cat(total_y, 0)
        
        if args.label_type == "dimensional":
            ccc = utils.CCC_loss(total_pred, total_y)            
            lm.add_torch_stat("dev_aro", ccc[0])
            lm.add_torch_stat("dev_dom", ccc[1])
            lm.add_torch_stat("dev_val", ccc[2])
        elif args.label_type == "categorical":
            if args.label_learning == "hard-label":
                loss = utils.CE_category(total_pred, total_y)
            elif args.label_learning == "soft-label":
                loss = utils.SCE_category(total_pred, total_y)
            elif args.label_learning == "multi-label":
                loss = utils.BCE_category(total_pred, total_y)
            elif args.label_learning == "distribution-label":
                loss = utils.KLD_category(total_pred, total_y)
            # acc = utils.calc_acc(total_pred, total_y)
            acc = utils.macro_f1(total_pred, total_y)
            lm.add_torch_stat("dev_loss", loss)
            lm.add_torch_stat("dev_acc", acc)


        lm.print_stat()
        if args.label_type == "dimensional":
            dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
        elif args.label_type == "categorical":
            dev_loss = lm.get_stat("dev_loss")
            tr_loss = lm.get_stat("train_loss")
            losses_dev.append(dev_loss)
            losses_train.append(tr_loss)
        if min_loss > dev_loss:
            min_epoch = epoch
            min_loss = dev_loss
        
        if float(dev_loss) < float(temp_dev):
            temp_dev = float(dev_loss)
            print('better dev loss found:' + str(float(dev_loss)) + ' saving model')
            modelWrapper.save_model(epoch)
    print("Save",end=" ")
    print(min_epoch, end=" ")
    print("")

    with open(model_path+'/train_loss.txt', 'w') as f:
        for item in losses_train:
            f.write("%s\n" % item)
    
    with open(model_path+'/dev_loss.txt', 'w') as f:
        for item in losses_dev:
            f.write("%s\n" % item)

    
    print("Loss",end=" ")
    if args.label_type == "dimensional":
        print(3.0-min_loss, end=" ")
    elif args.label_type == "categorical":
        print(min_loss, end=" ")
    print("")
    modelWrapper.save_final_model(min_epoch, remove_param=False)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--conf_path',
        default="config/conf.json",
        type=str)

    # Data Arguments
    parser.add_argument(
        '--corpus_type',
        default="podcast_v1.7",
        type=str)
    parser.add_argument(
        '--model_type',
        default="wav2vec2",
        type=str)
    parser.add_argument(
        '--label_type',
        choices=['dimensional', 'categorical'],
        default='categorical',
        type=str)

    # Chunk Arguments
    parser.add_argument(
        '--use_chunk',
        default=False,
        type=str2bool)
    parser.add_argument(
        '--chunk_hidden_dim',
        default=256,
        type=int)
    parser.add_argument(
        '--chunk_window',
        default=50,
        type=int)
    parser.add_argument(
        '--chunk_num',
        default=11,
        type=int)
    
    # Model Arguments
    parser.add_argument(
        '--model_path',
        default=None,
        type=str)
    parser.add_argument(
        '--output_num',
        default=4,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--epochs',
        default=100,
        type=int)
    parser.add_argument(
        '--lr',
        default=1e-5,
        type=float)
    
     # Label Learning Arguments
    parser.add_argument(
        '--label_learning',
        default="multi-label",
        type=str)

    parser.add_argument(
        '--corpus',
        default="USC-IEMOCAP",
        type=str)
    parser.add_argument(
        '--num_classes',
        default="four",
        type=str)
    parser.add_argument(
        '--label_rule',
        default="M",
        type=str)
    parser.add_argument(
        '--partition_number',
        default="1",
        type=str)
    parser.add_argument(
        '--data_mode',
        default="primary",
        type=str)



    args = parser.parse_args()

    # Call main function
    main(args)
