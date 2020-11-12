import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import math
import numpy as np
import os
import pickle
import random
import torch
import torch.nn.functional as F

from decimal import Decimal
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_utils import MetaFolderTwo, split_meta_both, r_at_k, np_transform, collate_recipe, mk_dataloader_1, mk_dataloader_2
from models import AEMetaModel, MergedMetaModel, MetaModel, TextEncoder, ImageEncoder, ImageClf, TextClf


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # First get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


dark_rainbow = cmap_map(lambda x: x*0.75, plt.cm.gist_rainbow)


def cos_loss(out1, out2, ys=None, margin=None):
    return 1 - F.cosine_similarity(out1, out2).mean()


def tri_loss(out1, out2, ys, margin=0.1):
    '''
    Args:
        out1: (batch_size, emb_dim)
        out2: (batch_size, emb_dim)
    '''
    out1 = out1/(torch.norm(out1,2, dim=1)[:, None])
    out2 = out2/(torch.norm(out2,2, dim=1)[:, None])
    y_counts = {}
    o1s = {}
    o2s = {}
    for o1, o2, y in zip(out1, out2, ys):
        y = y.cpu().item()
        if y in y_counts:
            y_counts[y] += 1
            o1s[y] += o1
            o2s[y] += o2
        else:
            y_counts[y] = 1
            o1s[y] = o1
            o2s[y] = o2
    avg_1s = []
    avg_2s = []
    for y in o1s:
        o1 = o1s[y] / y_counts[y]
        o2 = o2s[y] / y_counts[y]
        avg_1s.append(o1)
        avg_2s.append(o2)
    avg_1s = torch.stack(avg_1s)
    avg_2s = torch.stack(avg_2s)
    mat = torch.einsum('ij,kj->ik', avg_1s, avg_2s)
    diag = torch.diag(mat)
    return torch.sum(F.relu(margin + mat - diag.repeat(len(diag), 1)))/len(out1)


def print_log(s, log_path):
    print(s)
    with open(log_path, 'a+') as ouf:
        ouf.write("%s\n" % s)


def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def get_optimizer(net, lr, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser('few shot alignment with cos loss')

parser.add_argument('--seed', type=int, help='random seed', default=0)
parser.add_argument('--iseed', type=int, help='idx dict random seed', default=-1)
parser.add_argument('--load', action='store_true', help='load from ckpt', default=False)
parser.add_argument('--no-save', action='store_true', help='dont save model', default=False)
parser.add_argument('--no-eval', action='store_true', help='dont evaluate', default=False)
parser.add_argument('--print-train', action='store_true', help='print train metrics', default=False)
parser.add_argument('--verbose', action='store_true', help='print extra info', default=False)
parser.add_argument('--cuda', default=0, type=int, help='cuda device')
parser.add_argument('--num-workers', default=4, type=int, help='cuda device')
parser.add_argument('--margin', default=0.1, type=float, help='margin in loss fn')
parser.add_argument('--tfr', default=0.5, type=float, help='teacher forcing ratio')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--batch-size', default=256, type=int, help='number of epochs')
parser.add_argument('--mode', default="18", type=str, help='mode')

parser.add_argument('--no-meta-1', action='store_true', help='clf1 with no meta learning', default=False)
parser.add_argument('--no-meta-2', action='store_true', help='clf2 with no meta learning', default=False)
parser.add_argument('--reptile1', action='store_true', help='unimodal meta learning 1', default=False)
parser.add_argument('--reptile2', action='store_true', help='unimodal meta learning 2', default=False)
parser.add_argument('--train1', action='store_true', help='train test modality', default=False)
parser.add_argument('--ae', action='store_true', help='autoencoder', default=False)
parser.add_argument('--no-pre', action='store_true', help='dont record untrained model performance', default=False)
parser.add_argument('--do-super', action='store_true', help='supervised alignment', default=False)
parser.add_argument('--merge', action='store_true', help='same encoder for both modalities', default=False)
parser.add_argument('--merge0', action='store_true', help='merge but no metatrain test modality', default=False)
# NOTE merge & merge0 not applicable to recipe task

# few shot args
parser.add_argument('-n', '--classes', default=5, type=int, help='classes in base-task (N-way)')
parser.add_argument('--train-shots', default=5, type=int, help='(train) shots per class (K-shot)')
parser.add_argument('--meta-iterations', default=100000, type=int, help='number of meta iterations')
parser.add_argument('--start-meta-iteration', default=0, type=int, help='start iteration')
parser.add_argument('--iterations', default=5, type=int, help='number of base iterations')
parser.add_argument('--test-iterations', default=50, type=int, help='number of base iterations')
parser.add_argument('--train-align-batch', default=4, type=int, help='minibatch size in alignment metatrain task')
parser.add_argument('--batch', default=4, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=1., type=float, help='meta learning rate')
parser.add_argument('--lr-clf', default=1e-4, type=float, help='base learning rate')
parser.add_argument('--lr-align', default=1e-4, type=float, help='base learning rate')
parser.add_argument('--train', default=0.7, type=float, help='Percentage of train')
# parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
parser.add_argument('--validate-every', default=100, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--eval-tasks', default=-1, type=int, help='Number of eval tasks')
parser.add_argument('-l', default='tri', type=str, help='loss fn')

args = parser.parse_args()

if args.l == 'tri':
    align_loss = tri_loss
else:
    align_loss = cos_loss
ae_loss = nn.MSELoss()

cuda_device = args.cuda


def Variable_(tensor, *args_, **kwargs):
    if type(tensor) in (list, tuple):
        return [Variable_(t, *args_, **kwargs) for t in tensor]
    if isinstance(tensor, dict):
        return {key: Variable_(v, *args_, **kwargs) for key, v in tensor.items()}
    variable = Variable(tensor, *args_, **kwargs)
    variable = variable.cuda(cuda_device)
    return variable


def train_clf_2(net, loss_fn, optimizer, train_iter, iterations):
    losses = []
    net.train()
    for iteration in range(iterations):
        _, _, x2, base_y = Variable_(next(train_iter))
        prediction = net.forward2(x2)
        loss = loss_fn(prediction, base_y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(losses)


def train_align(net, loss_fn, optimizer, train_iter, iterations, margin=0.1):
    losses = []
    net.train()
    for iteration in range(iterations):
        x1, y1, x2, base_y = Variable_(next(train_iter))
        o1, o2 = net.forward_align(x1, x2)
        loss = loss_fn(o1, o2, base_y, margin=margin)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(losses)


def train_ae(net, loss_fn, optimizer, train_iter, iterations, teacher_forcing_ratio=0.5):
    losses = []
    net.train()
    for iteration in range(iterations): 
        x, _, _, _ = Variable_(next(train_iter))
        prediction = net.forward_ae(x, teacher_forcing_ratio)
        loss = loss_fn(x.squeeze(1), prediction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def train_clf_1(net, loss_fn, optimizer, train_iter, iterations, print_train=False):
    if print_train:
        print('-------')
    losses = []
    correct = 0
    total_samples = 0
    net.train()
    for iteration in range(iterations):
        x1, _, _, base_y = Variable_(next(train_iter))
        out = net.forward1(x1)
        loss = loss_fn(out, base_y)
        if print_train:
            pred = out.data.max(1, keepdim=True)[1]
            matchings = pred.eq(base_y.data.view_as(pred).type(torch.cuda.LongTensor))
            correct = correct + matchings.sum()
            total_samples = total_samples + len(base_y)
            print(loss)
            # print(matchings.sum())
            # print(iteration, ' ', loss, ' ', loss.item())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if print_train:
        acc = float(correct)/total_samples
        return np.mean(losses), acc
    else:
        return np.mean(losses)


def train_merge_clf(net, loss_fn, optimizer, train_iter, iterations):
    losses = []
    accuracies = []
    net.train()
    for iteration in range(iterations):
        x1, _, x2, base_y = Variable_(next(train_iter))
        prediction = net.forward1(x1)
        loss = loss_fn(prediction, base_y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = net.forward2(x2)
        loss = loss_fn(prediction, base_y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(losses)


def eval_align(net, loss_fn, test_iter, iterations, margin=0.1):
    losses = []
    net.eval()
    with torch.no_grad():
        for iteration in range(iterations):
            x1, y1, x2, base_y = Variable_(next(test_iter))
            o1, o2 = net.forward_align(x1, x2)
            loss = loss_fn(o1, o2, base_y, margin=margin)
            losses.append(loss.item())
    return np.mean(losses)


def eval_align_r(net, loss_fn, test_iter, iterations, margin=0.1):
    losses = []
    net.eval()
    with torch.no_grad():
        y1s = []
        all_transformed_audio_embs = []
        all_transformed_text_embs = []
        for iteration in range(iterations):
            x1, y1, x2, base_y = Variable_(next(test_iter))
            y1s.append(y1.cpu().numpy())
            o1, o2 = net.forward_align(x1, x2)
            all_transformed_audio_embs.append(o1.cpu().numpy())
            all_transformed_text_embs.append(o2.cpu().numpy())
            loss = loss_fn(o1, o2, base_y, margin=margin)
            losses.append(loss.item())
        y1s = np.concatenate(y1s, axis=0)
        all_transformed_audio_embs = np.concatenate(all_transformed_audio_embs, axis=0)
        all_transformed_text_embs = np.concatenate(all_transformed_text_embs, axis=0)
    return np.mean(losses), y1s, all_transformed_audio_embs, all_transformed_text_embs


def eval_clf_2(net, loss_fn, test_iter, iterations):
    losses = []
    correct = 0
    total_samples = 0
    net.eval()
    with torch.no_grad():
        for iteration in range(iterations):
            _, _, x2, base_y = Variable_(next(test_iter))
            out = net.forward2(x2)
            loss = loss_fn(out, base_y)
            losses.append(loss.item())
            pred = out.data.max(1, keepdim=True)[1]
            matchings = pred.eq(base_y.data.view_as(pred).type(torch.cuda.LongTensor))
            correct = correct + matchings.sum()
            total_samples = total_samples + x2.size()[0]   
    val_acc = float(correct)/total_samples
    return np.mean(losses), val_acc


def eval_clf_1(net, loss_fn, test_iter, iterations):
    losses = []
    correct = 0
    total_samples = 0
    net.eval()
    with torch.no_grad():
        for iteration in range(iterations):
            x1, _, _, base_y = Variable_(next(test_iter))
            out = net.forward1(x1)
            loss = loss_fn(out, base_y)
            losses.append(loss.item())
            pred = out.data.max(1, keepdim=True)[1]
            matchings = pred.eq(base_y.data.view_as(pred).type(torch.cuda.LongTensor))
            correct = correct + matchings.sum()
            total_samples = total_samples + x1.size()[0]
    val_acc = float(correct)/total_samples
    return np.mean(losses), val_acc


def train_no_meta(net, loss_fn, loader, optimizer, args):
    net.train()
    train_losses = []
    preds = []
    trues = []
    for batch_idx, (train_x, train_y) in enumerate(loader):
        train_x = train_x.cuda(args.cuda)
        train_y = train_y.cuda(args.cuda)
        train_out = net(train_x)
        loss = loss_fn(train_out, train_y)
        train_losses.append(loss.item())
        net.zero_grad()
        loss.backward()
        optimizer.step()
        train_pred = train_out.data.max(1, keepdim=True)[1].cpu().numpy()[:,0]
        preds.append(train_pred)
        trues.append(train_y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    metrics_dict = {}
    metrics_dict["accuracy"] = accuracy_score(trues, preds)
    # metrics_dict["precision"] = precision_score(trues, preds, average="macro")
    # metrics_dict["recall"] = recall_score(trues, preds, average="macro")
    # metrics_dict["f1_score"] = f1_score(trues, preds, average="macro")
    return np.mean(train_losses), metrics_dict


def eval_no_meta(net, loss_fn, loader, args):
    losses = []
    preds = []
    trues = []
    net.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda(args.cuda)
            y = y.cuda(args.cuda)
            out = net(x)
            loss = loss_fn(out, y)
            losses.append(loss.item())
            pred = out.data.max(1, keepdim=True)[1].cpu().numpy()[:,0]
            preds.append(pred)
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    metrics_dict = {}
    metrics_dict["accuracy"] = accuracy_score(trues, preds)
    metrics_dict["precision"] = precision_score(trues, preds, average="macro")
    metrics_dict["recall"] = recall_score(trues, preds, average="macro")
    metrics_dict["f1_score"] = f1_score(trues, preds, average="macro")
    metrics_dict["confusion_matrix"] = confusion_matrix(trues, preds, normalize="true")
    return np.mean(losses), metrics_dict


random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

curr_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(curr_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
mlr_str = format_e(Decimal(args.meta_lr))
lrc_str = format_e(Decimal(args.lr_clf))
lra_str = format_e(Decimal(args.lr_align))
expt_str = '-%d-%d' % (args.iterations, args.test_iterations)
if args.do_super:
    expt_str += '-sup'
if args.ae:
    expt_str += '-ae'
if args.merge:
    expt_str += '-merge'
if args.merge0:
    expt_str += '-merge0'
if args.reptile1:
    expt_str += '-reptile1_%s' % args.mode
if args.reptile2:
    expt_str += '-reptile2'
if args.no_meta_1:
    expt_str += '-no_meta_1_%s' % args.mode
if args.no_meta_2:
    expt_str += '-no_meta_2'
if args.eval_tasks == -1:
    log_path = os.path.join(log_dir, 'log_k-%d_mlr-%s_lrc-%s_lra-%s_tab-%d_s-%d-%d%s.txt' % (args.train_shots, mlr_str, lrc_str, lra_str, args.train_align_batch, args.seed, args.iseed, expt_str))
    ckpt_path = os.path.join(log_dir, 'meta_k-%d_mlr-%s_lrc-%s_lra-%s_tab-%d_s-%d-%d%s.ckpt' % (args.train_shots, mlr_str, lrc_str, lra_str, args.train_align_batch, args.seed, args.iseed, expt_str))
else:
    log_path = os.path.join(log_dir, 'log_k-%d_mlr-%s_lrc-%s_lra-%s_tab-%d_e-%d_s-%d-%d%s.txt' % (args.train_shots, mlr_str, lrc_str, lra_str, args.train_align_batch, args.eval_tasks, args.seed, args.iseed, expt_str))
    ckpt_path = os.path.join(log_dir, 'meta_k-%d_mlr-%s_lrc-%s_lra-%s_tab-%d_e-%d_s-%d-%d%s.ckpt' % (args.train_shots, mlr_str, lrc_str, lra_str, args.train_align_batch, args.eval_tasks, args.seed, args.iseed, expt_str))
print_log(log_path, log_path)

if args.no_meta_1:
    num_classes, train_loader = mk_dataloader_1('train', batch_size=args.batch_size, num_workers=args.num_workers)
    _, test_loader = mk_dataloader_1('test', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
elif args.no_meta_2:
    num_classes, train_loader = mk_dataloader_2('train', batch_size=args.batch_size, num_workers=args.num_workers)
    _, test_loader = mk_dataloader_2('test', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
else:
    idx_dir = '../data/recipe/idxs'

    collate_fn = collate_recipe
    all_meta = MetaFolderTwo(transform_x1=np_transform, transform_x2=np_transform)

    if args.do_super:
        align_train, align_val, align_test, super_train = split_meta_both(all_meta, train=args.train, seed=args.iseed, mk_super=True, verbose=args.verbose, collate_fn=collate_recipe)
    else:
        align_train, align_val, align_test = split_meta_both(all_meta, train=args.train, seed=args.iseed, dept=args.ae, verbose=args.verbose, collate_fn=collate_recipe)

cross_entropy = nn.CrossEntropyLoss()

fc_dim = 128
if args.ae:
    meta_net = AEMetaModel(fc_dim, args.classes, args.cuda)
elif args.merge:
    # NOTE: not supported for recipe task
    meta_net = MergedMetaModel(fc_dim, args.classes, vocab_size, emb_mat, args.cuda)
elif args.no_meta_1:
    meta_net = ImageClf(fc_dim, num_classes, mode=args.mode)
elif args.no_meta_2:
    meta_net = TextClf(fc_dim, num_classes, mode=args.mode)
else:
    enc1 = ImageEncoder(fc_dim)
    enc2 = TextEncoder(fc_dim)
    meta_net = MetaModel(enc1, enc2, fc_dim, args.classes, args.cuda)
if torch.cuda.is_available():
    meta_net = meta_net.cuda(args.cuda)
# meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=args.meta_lr)
meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr)
audio_clf_state = None
text_clf_state = None
align_state = None
if args.do_super:
    super_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.lr_align)
elif args.no_meta_1 or args.no_meta_2:
    no_meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.lr_clf)

if args.load and os.path.exists(ckpt_path):
    loaded_states = torch.load(ckpt_path)
    meta_net.load_state_dict(loaded_states['model'])
    meta_optimizer.load_state_dict(loaded_states['optim'])

if not args.no_meta_1 and not args.no_meta_2:
    idx_dict_path = os.path.join(idx_dir, 'idx_dict_%d_%d_%d_%d.npy' % (args.classes, args.train_shots, args.eval_tasks, args.iseed))
    print(idx_dict_path)
    idx_dict = np.load(idx_dict_path, allow_pickle=True)
    idx_dict = idx_dict[()]

best_metric = 0.0
best_iter = 0

# for early stopping
stop_iters = 25*args.validate_every

# -------------------
# pretrain meta evaluation

# evaluate clf 1 (image for recipe, audio for wld)
if not args.no_pre:
    if args.no_meta_1 or args.no_meta_2:
        loss, metrics = eval_no_meta(meta_net, cross_entropy, test_loader, args)
        prefix_str = '-1 clf 1\t'
        print('                trainl traina loss   acc    prec   rec    f1')
        metrics_str = '------ ------ %.4f %.4f %.4f %.4f %.4f' % (loss, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"])
        print_log(prefix_str+metrics_str, log_path)
    else:
        metrics = []
        # for (meta_dataset, mode) in [(align_train, 'train'), (align_val, 'val'), (align_test, 'test')]:
        for (meta_dataset, mode) in [(align_train, 'train'), (align_test, 'test')]:
            mode = 'meta_'+mode
            curr_idx_dict = idx_dict[mode]
            # print(curr_idx_dict)
            meta_losses = []
            meta_accuracies = []
            for task_idx_dict in curr_idx_dict:
                character_indices = task_idx_dict['character_indices']
                all_curr_idxs = task_idx_dict['all_curr_idxs']
                new_train_idxs = task_idx_dict['new_train_idxs']
                new_test_idxs = task_idx_dict['new_test_idxs']
                train, val = meta_dataset.get_task_split(character_indices, all_curr_idxs, new_train_idxs, new_test_idxs, train_K=args.train_shots)
                train_iter = make_infinite(DataLoader(train, args.batch, collate_fn=collate_fn, num_workers=args.num_workers))
                val_iter = make_infinite(DataLoader(val, args.batch, collate_fn=collate_fn, num_workers=args.num_workers))
                net = meta_net.clone()
                optimizer = get_optimizer(net, args.lr_clf)
                num_iter = int(math.ceil(len(val)/args.batch))
                if not args.reptile2:
                    if args.print_train:
                        meta_train_loss, meta_train_acc = train_clf_1(net, cross_entropy, optimizer, train_iter, args.test_iterations, args.print_train)
                        print(meta_train_loss, meta_train_acc)
                    else:
                        meta_train_loss = train_clf_1(net, cross_entropy, optimizer, train_iter, args.test_iterations, args.print_train)
                    meta_loss, meta_accuracy = eval_clf_1(net, cross_entropy, val_iter, num_iter)
                else:
                    meta_train_loss = train_clf_2(net, cross_entropy, optimizer, train_iter, args.test_iterations)
                    meta_loss, meta_accuracy = eval_clf_2(net, cross_entropy, val_iter, num_iter)
                meta_losses.append(meta_loss)
                meta_accuracies.append(meta_accuracy)
            metrics.append(meta_train_loss)
            metrics.append(np.mean(meta_losses))
            metrics.append(np.mean(meta_accuracies))
        prefix_str = '-1 clf 1\t'
        metrics_str = ' '.join(['%.4f' % f for f in metrics])
        print_log(prefix_str+metrics_str, log_path)

# -------------------
# Main loop

if args.do_super:
    super_train = make_infinite(super_train)

if args.no_meta_1 or args.no_meta_2:
    best_epoch = -1
    for epoch in range(args.epochs):
        train_loss, train_metrics = train_no_meta(meta_net, cross_entropy, train_loader, no_meta_optimizer, args)
        loss, metrics = eval_no_meta(meta_net, cross_entropy, test_loader, args)
        train_acc = train_metrics["accuracy"]
        metric = metrics["f1_score"]
        if metric > best_metric:
            conf_mat = metrics["confusion_matrix"]/np.max(metrics["confusion_matrix"])*255
            conf_mat = conf_mat.astype('uint8')
            im = Image.fromarray(conf_mat)
            im.save('conf_mat.png')
        prefix_str = '%02d clf 1\t' % epoch
        metrics_str = '%.4f %.4f %.4f %.4f %.4f %.4f %.4f' % (train_loss, train_acc, loss, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"])
        print_log(prefix_str+metrics_str, log_path)
else:
    suffix_str = ' <'
    for meta_iteration in range(args.start_meta_iteration, args.meta_iterations):
        # Update learning rate
        '''TODO i commented this out
        if not args.do_super:
            meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iterations))
            set_learning_rate(meta_optimizer, meta_lr)
        '''

        if args.reptile1 or args.train1:
            net = meta_net.clone()
            optimizer = get_optimizer(net, args.lr_clf, audio_clf_state)
            train = align_train.get_random_task(args.classes, args.train_shots, is_align=False)
            # train = align_train.get_dummy_task(args.classes, args.train_shots, is_align=False)
            train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers))
            if args.print_train:
                loss, acc = train_clf_1(net, cross_entropy, optimizer, train_iter, args.iterations, print_train=args.print_train)
            else:
                loss = train_clf_1(net, cross_entropy, optimizer, train_iter, args.iterations, print_train=args.print_train)
                # print(loss) # TODO toggle comment to debug
            audio_clf_state = optimizer.state_dict()
            meta_net.point_grad_to(net)
            meta_optimizer.step()
            if args.print_train:
                print(loss, acc)

        if args.merge:
            net = meta_net.clone()
            optimizer = get_optimizer(net, args.lr_clf, audio_clf_state)
            train = align_train.get_random_task(args.classes, args.train_shots, is_align=False)
            train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers))
            loss = train_merge_clf(net, cross_entropy, optimizer, train_iter, args.iterations)
            audio_clf_state = optimizer.state_dict()
            meta_net.point_grad_to(net)
            meta_optimizer.step()

        if args.ae:
            net = meta_net.clone()
            optimizer = get_optimizer(net, args.lr_clf, text_clf_state)
            train = align_train.get_random_task(args.classes, args.train_shots)
            train_iter = make_infinite(DataLoader(train, args.train_align_batch, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers))
            loss = train_ae(net, ae_loss, optimizer, train_iter, args.iterations, teacher_forcing_ratio=args.tfr)
            text_clf_state = optimizer.state_dict()
            meta_net.point_grad_to(net)
            meta_optimizer.step()

        # train clf 2
        if not args.reptile1 and not args.ae and not args.merge:
            net = meta_net.clone()
            optimizer = get_optimizer(net, args.lr_clf, text_clf_state)
            train = align_train.get_random_task(args.classes, args.train_shots, is_align=False)
            train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers))
            loss = train_clf_2(net, cross_entropy, optimizer, train_iter, args.iterations)
            text_clf_state = optimizer.state_dict()
            meta_net.point_grad_to(net)
            meta_optimizer.step()
            if args.verbose:
                print(loss)

        # train alignment
        if args.do_super:
            losses = []
            meta_net.train()
            for iteration in range(args.iterations):
                x1, y1, x2, y2 = Variable_(next(super_train))
                o1, o2 = meta_net.forward_align(x1, x2)
                loss = align_loss(o1, o2, y1, margin=args.margin)
                losses.append(loss.cpu().item())
                super_optimizer.zero_grad()
                loss.backward()
                super_optimizer.step()
        elif not args.reptile1 and not args.ae and not args.merge and not args.merge0 and not args.reptile2:
            net = meta_net.clone()
            optimizer = get_optimizer(net, args.lr_align, align_state)
            train = align_train.get_random_task(args.classes, args.train_shots, is_align=True)
            train_iter = make_infinite(DataLoader(train, args.train_align_batch, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers))
            loss = train_align(net, align_loss, optimizer, train_iter, args.iterations, margin=args.margin)
            align_state = optimizer.state_dict()
            meta_net.point_grad_to(net)
            meta_optimizer.step()

        if meta_iteration % args.validate_every == 0 and not args.no_eval:
            # evaluate clf 1
            metrics = []
            # for (meta_dataset, mode) in [(align_test, 'test')]: # [(align_train, 'train'), (align_val, 'val'), (align_test, 'test')]:
            for (meta_dataset, mode) in [(align_train, 'train'), (align_test, 'test')]:
                mode = 'meta_'+mode
                curr_idx_dict = idx_dict[mode]
                meta_losses = []
                meta_accuracies = []
                for task_idx_dict in curr_idx_dict:
                    character_indices = task_idx_dict['character_indices']
                    all_curr_idxs = task_idx_dict['all_curr_idxs']
                    new_train_idxs = task_idx_dict['new_train_idxs']
                    new_test_idxs = task_idx_dict['new_test_idxs']
                    train, val = meta_dataset.get_task_split(character_indices, all_curr_idxs, new_train_idxs, new_test_idxs, train_K=args.train_shots)
                    train_iter = make_infinite(DataLoader(train, args.batch, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers))
                    val_iter = make_infinite(DataLoader(val, args.batch, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers))
                    net = meta_net.clone()
                    optimizer = get_optimizer(net, args.lr_clf)
                    num_iter = int(math.ceil(len(val)/args.batch))
                    if not args.reptile2:
                        meta_train_loss = train_clf_1(net, cross_entropy, optimizer, train_iter, args.test_iterations)
                        meta_loss, meta_accuracy = eval_clf_1(net, cross_entropy, val_iter, num_iter)
                    else:
                        meta_train_loss = train_clf_2(net, cross_entropy, optimizer, train_iter, args.test_iterations)
                        meta_loss, meta_accuracy = eval_clf_2(net, cross_entropy, val_iter, num_iter)
                    meta_losses.append(meta_loss)
                    meta_accuracies.append(meta_accuracy)
                metrics.append(meta_train_loss)
                metrics.append(np.mean(meta_losses))
                metrics.append(np.mean(meta_accuracies))
            prefix_str = '%02d clf 1\t' % meta_iteration
            metrics_str = ' '.join(['%.4f' % f for f in metrics])
            if metrics[-1] > best_metric:
                print_log(prefix_str+metrics_str+suffix_str, log_path)
                suffix_str = suffix_str+'<'
                best_metric = metrics[-1]
                best_iter = meta_iteration
                if not args.no_save:
                    if args.do_super:
                        torch.save({'model': meta_net.state_dict(), 'optim': super_optimizer.state_dict()}, ckpt_path)
                    else:
                        torch.save({'model': meta_net.state_dict(), 'optim': meta_optimizer.state_dict()}, ckpt_path)
            elif meta_iteration - best_iter > stop_iters:
                print_log(prefix_str+metrics_str, log_path)
                exit()
            else:
                print_log(prefix_str+metrics_str, log_path)
