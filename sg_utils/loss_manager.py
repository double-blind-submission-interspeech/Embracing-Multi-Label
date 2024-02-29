import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import torch.autograd as autograd
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

class LogManager:
    def __init__(self):
        self.log_book=dict()
    def alloc_stat_type(self, stat_type):
        self.log_book[stat_type] = []
    def alloc_stat_type_list(self, stat_type_list):
        for stat_type in stat_type_list:
            self.alloc_stat_type(stat_type)
    def init_stat(self):
        for stat_type in self.log_book.keys():
            self.log_book[stat_type] = []
    def add_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat)
    def add_torch_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat.detach().cpu().item())
    def get_stat(self, stat_type):
        result_stat = 0
        stat_list = self.log_book[stat_type]
        if len(stat_list) != 0:
            result_stat = np.mean(stat_list)
            result_stat = np.round(result_stat, 4)
        return result_stat

    def print_stat(self):
        for stat_type in self.log_book.keys():
            if len(self.log_book[stat_type]) == 0:
                continue
            stat = self.get_stat(stat_type)           
            print(stat_type,":",stat, end=' / ')
        print(" ")

def CCC_loss(pred, lab, m_lab=None, v_lab=None):
    """
    pred: (N, 3)
    lab: (N, 3)
    """
    m_pred = torch.mean(pred, 0, keepdim=True)
    m_lab = torch.mean(lab, 0, keepdim=True)

    d_pred = pred - m_pred
    d_lab = lab - m_lab

    v_pred = torch.var(pred, 0, unbiased=False)
    v_lab = torch.var(lab, 0, unbiased=False)

    corr = torch.sum(d_pred * d_lab, 0) / (torch.sqrt(torch.sum(d_pred ** 2, 0)) * torch.sqrt(torch.sum(d_lab ** 2, 0)))

    s_pred = torch.std(pred, 0, unbiased=False)
    s_lab = torch.std(lab, 0, unbiased=False)

    ccc = (2*corr*s_pred*s_lab) / (v_pred + v_lab + (m_pred[0]-m_lab[0])**2)    
    return ccc

def ladder_loss(decoder_h, encoder_h, layer_wise=False):
    assert len(decoder_h) == len(encoder_h)

    h_num = len(decoder_h)
    if layer_wise:
        total_loss=[torch.zeros(1).float().cuda() for h in range(h_num)]
    else:
        total_loss = torch.zeros(1).float().cuda()
    for h_idx in range(h_num):
        if layer_wise:
            total_loss[h_idx] += F.mse_loss(decoder_h[h_idx], encoder_h[h_num-1-h_idx])
        else:
            total_loss += F.mse_loss(decoder_h[h_idx], encoder_h[h_num-1-h_idx])
    return total_loss

def decoupled_ladder_loss(decoder_he, encoder_he, decoder_hr, encoder_hr):
    assert len(decoder_he) == len(encoder_he) == len(decoder_hr) == len(encoder_hr)

    h_num = len(decoder_he)
    total_loss = torch.zeros(1).float().cuda()
    for h_idx in range(h_num):
        if h_idx == h_num - 1:
            x = encoder_he[h_num-h_idx-1]
            recon_x = decoder_he[h_idx]+decoder_hr[h_idx]
            total_loss += F.mse_loss(recon_x, x)
        else:
            total_loss += F.mse_loss(decoder_he[h_idx], encoder_hr[h_num-h_idx-1])
            total_loss += F.mse_loss(decoder_he[h_idx], encoder_hr[h_num-h_idx-1])
    return total_loss


def orthogonal_loss(eh, rh, eps=0.0):
    batch_size = eh.size(0)
    out = torch.zeros(1).cuda()
    
    for e, r in zip(eh, rh):
        len_e = torch.sqrt(torch.sum(torch.pow(e, 2)))
        len_r = torch.sqrt(torch.sum(torch.pow(r, 2)))
        out += torch.dot(e, r) / ((len_e*len_r)+eps)
    out /= batch_size
    out = torch.abs(out)
    return out

def MSE_emotion(pred, lab):
    aro_loss = F.mse_loss(pred[:][0], lab[:][0])
    dom_loss = F.mse_loss(pred[:][1], lab[:][1])
    val_loss = F.mse_loss(pred[:][2], lab[:][2])

    return [aro_loss, dom_loss, val_loss]


# For Hard-label learning
def CE_category(pred, lab):
    celoss = nn.CrossEntropyLoss()
    max_indx = torch.argmax(lab, dim=1)
    ce_loss = celoss(pred, max_indx)
    return ce_loss

# For Soft-label learning
def SCE_category(pred, lab):
    lsm = F.log_softmax(pred, -1)
    loss = -(lab * lsm).sum(-1)
    return loss.mean()

def class_balanced_softmax_cross_entropy_with_softtarget(input,target,weights,reduction='mean'):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    :weights from the training set: [0.2139, 0.3445, 0.4665, 3.3741, 0.3337, 2.2446, 0.2398, 1.3928, 0.3900]
    :original batch loss: [3.0727, 2.7817, 1.7426, 3.5614, 2.9689, 4.5142, 3.8597, 2.8744]
    :weighted batch loss: [1.2815, 0.7407, 0.5930, 1.4853, 3.0804, 1.5423, 1.9164, 1.1688]
    """

    weights = weights.unsqueeze(0)
    weights = weights.repeat(target.shape[0],1) * target
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,target.shape[1]) # Dim = (batch,K_emotions)

    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(weights*target.view(target.shape[0], -1) * logprobs, dim=1)

    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')

# For Multi-label learning
def BCE_category(pred,lab):
    bceloss = nn.BCEWithLogitsLoss()
    p = pred.detach()
    total_num = p.size()[1]
    bar_thre = 0.05/(total_num-1)
    # Multiple-hot vector
    target = lab.detach()
    target_multilabel = torch.zeros(target.size(), device=torch.device('cuda'))
    target_multilabel[target>bar_thre]=1
    bce_loss = bceloss(pred,target_multilabel)
    # if we want the proabilities of preditions, pass the pred into the Sigmoid funtion.
    return bce_loss

# For Distribution-label learning
def KLD_category(pred, lab):

    log_pred = F.log_softmax(pred)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    return kl_loss(log_pred, lab)

def NLL_category(pred, lab):
    return nn.NLLLoss()(pred, lab)

def calc_err(pred, lab):
    p = pred.detach()
    t = lab.detach()
    total_num = p.size()[0]
    ans = torch.argmax(p, dim=1)
    tar = torch.argmax(t, dim=1)
    #print('ans',ans.shape)
    #print('tar',tar.shape)
    corr = torch.sum((ans==tar).long())
    err = (total_num-corr) / total_num
    return err

def calc_acc(pred, lab):
    err = calc_err(pred, lab)
    return 1.0 - err

def macro_f1(pred, lab):
    p = pred.detach()
    t = lab.detach()
    k_thresold = 1/t.shape[1]
    prediction_distribution = torch.nn.functional.softmax(p,dim=1)
    preditions_binary = torch.where(prediction_distribution> k_thresold, 1.0, 0.0)
    labels_binary = torch.where(t > k_thresold, 1.0, 0.0)
    reprot_dict = classification_report(labels_binary.cpu(), preditions_binary.cpu(),output_dict=True)
    macrof1 = reprot_dict['macro avg']['f1-score']
    return torch.from_numpy(np.array(macrof1))

def self_entropy(log_prob):
    prob = torch.exp(log_prob)
    b = prob * torch.log2(prob)
    b = torch.mean(-1.0 * b.sum(dim=1))
    return b

def calc_rank_loss(pair_set, rank_scores):
    batch_len = len(pair_set)
    loss = torch.zeros(1).cuda()
    for higher_idx, lower_idx in pair_set:
        score_higher = rank_scores[higher_idx]
        score_lower = rank_scores[lower_idx]

        loss += torch.log(1+torch.exp(-1*(score_higher-score_lower)))
    loss /= batch_len
    return loss

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand_like(real_data)
    # alpha = torch.ones_like(real_data)
    # alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients + 1e-16
    gradient_penalty = ((gradients.norm(2, dim=(1,2)) - 1) ** 2).mean()
    # gradient_penalty = gradient_penalty.float()
    return gradient_penalty

def calc_moving_average(pre_ma, cur_ma, gamma=0.99):
    cur_val = torch.mean(cur_ma)
    result_ma = gamma * pre_ma + (1-gamma) * cur_val
    return result_ma