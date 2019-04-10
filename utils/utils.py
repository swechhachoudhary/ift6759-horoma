from time import time
from data.dataset import HoromaDataset, OriginalHoromaDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils.constants import Constants
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

def get_acc(output, label):
    pred = torch.argmax(output, dim=1, keepdim=False)
    correct = torch.mean((pred == label).type(torch.FloatTensor))
    return correct

def get_split_indices(targets, overlapped=False, split=0.7):
    """
    Compute splits by separating regions in labeled training and validation sets while also
    to the extent possible preserving class balance. Priority is given to separating regions.

    :param targets: Numpy array of labels.
    :param overlapped: Choice of validation set.
    :param split: Approximate percentage of data in labeled training set.
    :return: training indices, validation indices
    """
    if overlapped:
        regions = np.loadtxt(Constants.REGION_ID_PATH_OVERLAPPED)
    else:
        regions = np.loadtxt(Constants.REGION_ID_PATH)

    idx_map = _map_to_idx(regions)
    region_map = _map_to_regions(regions, targets)
    counts_map = _map_to_counts(regions, targets, idx_map)

    train_indices = []

    unique_targets, counts = np.unique(targets, return_counts=True)
    n_train_remain = counts // (1 / split)
    seen_regions = set([])

    for target in unique_targets:
        n_train = n_train_remain[target]
        while n_train > 0:
            region = _next_region_for_target(region_map, target, seen_regions)
            seen_regions.add(region)
            target_counts = counts_map[region]
            n_train_remain -= target_counts
            train_indices.extend(idx_map[region])
            n_train = n_train_remain[target]

    valid_indices = [i for i in range(len(targets)) if i not in train_indices]

    return train_indices, valid_indices


def _next_region_for_target(region_map, target, seen_regions):
    """
    Find an unexplored region which contains a sample for the given target.

    :param region_map: Pre-computed dictionary of target --> associated regions.
    :param target: Label in question.
    :param seen_regions: Set of previously explored regions. We want to avoid returning these twice.
    :return: Region id.
    """
    while True:
        region = region_map[target].pop()
        if region not in seen_regions:
            break

    return region


def _map_to_counts(regions, targets, idx_map):
    """
    Compute a mapping associating each region to an array of counts for each class.

    :param regions: array of region ids.
    :param targets: array of targets.
    :param idx_map: mapping for region --> indices
    :return: Dictionary associating each region to an array of counts for each class.
    """
    num_targets = len(np.unique(targets))
    mapping = {region: np.zeros(num_targets) for region in np.unique(regions)}
    for region in mapping:
        region_targets = targets[idx_map[region]]
        for t in region_targets:
            mapping[region][t] += 1
    return mapping


def _map_to_regions(regions, targets):
    """
    Helper function computing a mapping from labels to association regions.

    :param regions: Array of region ids.
    :param targets: Array of targets
    :return: dict for mapping target --> regions.
    """
    pairs = zip(targets, regions)
    mapping = {target: [] for target in targets}
    for key, val in pairs:
        if val not in mapping[key]:
            mapping[key].append(val)
    return mapping


def _map_to_idx(regions):
    """
    Helper function associating regions with their indices (samples)in the file.

    :param regions: Array of region ids.
    :return: Indices corresponding to this region.
    """
    idx = {elem: [] for elem in np.unique(regions)}
    for i, id in enumerate(regions):
        idx[id].append(i)
    return idx


def load_datasets(datapath, train_subset, flattened=False, split="train_all"):
    """
    Load Horoma datasets from specified data directory.

    :type datapath: str
    :type flattened: bool
    :type train_subset: str
    """

    print("Loading datasets from ({}) ...".format(datapath), end=' ')
    start_time = time()
    dataset = HoromaDataset(
        datapath, split=split, subset=train_subset, flattened=flattened)

    print("Done in {:.2f} sec".format(time() - start_time))

    return dataset


def load_original_horoma_datasets(datapath, train_subset, flattened=False, overlapped=True):
    """
    Load Original Horoma datasets from specified data directory.
    Return unlabeled, labeled and validation sets

    :type datapath: str
    :type flattened: bool
    :type train_subset: str
    :type overlapped: bool
    """

    print("Loading datasets from ({}) ...".format(datapath), end=' ')
    start_time = time()
    if overlapped:
        unlabeled_trainset = OriginalHoromaDataset(datapath, split="train_overlapped", subset=train_subset, flattened=flattened)
        labeled_trainset = OriginalHoromaDataset(datapath, split="train_labeled_overlapped", flattened=flattened)
        labeled_validset = OriginalHoromaDataset(datapath, split="valid_overlapped", flattened=flattened)

    else:
        unlabeled_trainset = OriginalHoromaDataset(datapath, split="train", subset=train_subset, flattened=flattened)
        labeled_trainset = OriginalHoromaDataset(datapath, split="train_labeled", flattened=flattened)
        labeled_validset = OriginalHoromaDataset(datapath, split="valid", flattened=flattened)
        labeled_train_valid_set = OriginalHoromaDataset(datapath, split="train_labeled", flattened=flattened)
        labeled_train_valid_set.data = np.concatenate([labeled_train_valid_set.data, labeled_validset.data])
        labeled_train_valid_set.targets = np.concatenate([labeled_train_valid_set.targets, labeled_validset.targets])

    print("Done in {:.2f} sec".format(time() - start_time))
    return unlabeled_trainset, labeled_trainset, labeled_validset, labeled_train_valid_set

def return_images(data):
    all_embeddings=[]
    all_targets = []
    loader = DataLoader(data, batch_size = 32,shuffle=True)
    cuda = True if torch.cuda.is_available() else False
    labeled = True
    if loader.dataset.data.shape[0] >500 :
        labeled = False

    for imgs in loader:
       
        if labeled:
            (imgs, target) = imgs

        if cuda:
            data = Variable(imgs).cuda()
        else:
            data = Variable(imgs)       
        data =data.view(-1,3*32*32).cpu().data.numpy()
        for l in range(np.shape(data)[0]):
            all_embeddings.append(data[l])
            if labeled:
                all_targets.append(target[l].numpy()[0])

    return all_embeddings,all_targets

def assign_labels_to_clusters(model, data, labels_true):
    """
    Assign class label to each model cluster using labeled data.
    The class label is based on the class of majority samples within a cluster.
    Unassigned clusters are labeled as -1.
    """
    print("Assigning labels to clusters ...", end=' ')
    start_time = time()

    labels_pred = model.predict_cluster(data)
    labelled_clusters = []
    for i in range(model.n_clusters):
        idx = np.where(labels_pred == i)[0]
        if len(idx) != 0:
            labels_freq = np.bincount(labels_true[idx])
            labelled_clusters.append(np.argmax(labels_freq))
        else:
            labelled_clusters.append(-1)
    print("Done in {:.2f} sec".format(time() - start_time))

    return np.asarray(labelled_clusters)


def compute_metrics(y_true, y_pred):
    """ From the true and predicted labels, return the accuracy and the f1 score"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, f1


def eval_model_predictions(model, x, y_true, cluster_labels):
    """
    Predict labels and compare to true labels to compute the accuracy and f1 score
    """
    print("Evaluating model ...", end=' ')
    start_time = time()
    y_pred = cluster_labels[model.predict_cluster(x)]

    accuracy, f1 = compute_metrics(y_true, y_pred)

    print(
        "Done in {:.2f} sec | Accuracy: {:.2f} - F1: {:.2f}".format(time() - start_time, accuracy * 100, f1 * 100))

    return y_pred, accuracy, f1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight)
        # m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Bias') != -1:
        m.bias.data.fill_(0)
        

def train_nrm(net, train_loader,labeled_loader, eval_loader, num_epochs, configs,n_iterations):
    best_f1 = 0
    valid_accuracies = []
    f1_scores = []


    device = 'cuda'
    NO_LABEL = -1
    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    L2_loss = nn.MSELoss(size_average=False, reduce=False, reduction='mean').cuda()
    trainer = torch.optim.Adam(net.parameters(), configs['lr'][0], weight_decay=configs['decay'])

    best_valid_acc = 0
    iter_indx = 0

    for epoch in range(1,num_epochs):
        train_loss = 0; train_loss_xentropy = 0; train_loss_reconst = 0; train_loss_pn = 0; train_loss_kl = 0; train_loss_bnmm = 0
        correct = 0
        num_batch_train = 0

        # start with adam optimizer but switch sgd optimizer with exponential decay learning rate since epoch 20 
        if epoch == 200:
            sgd_lr = configs['lr'][1]
            decay_val = np.exp(np.log(sgd_lr / 0.0001) / (num_epochs - 2))
            sgd_lr = sgd_lr * decay_val
            trainer = torch.optim.SGD(net.parameters(), sgd_lr, weight_decay=configs['decay'])

        if epoch >= 150:
            for param_group in trainer.param_groups:
                param_group['lr'] = param_group['lr']/decay_val

        for param_group in trainer.param_groups:
            learning_rate = param_group['lr']


        # switch to train mode
        net.train()

    #     end = time.time()
        for i in range(int(n_iterations)):
    #         meters.update('data_time', time.time() - end)

            unsup_batch = next(iter(train_loader))
            sup_batch,target = next(iter(labeled_loader))

            # set up unlabeled input and labeled input with the corresponding labels
            input_unsup_var = torch.autograd.Variable(unsup_batch[0:(configs['batch_size'] - configs['labeled_batch_size'])]).to(device)
            input_sup_var = torch.autograd.Variable(sup_batch).to(device)
            target_sup_var = torch.autograd.Variable(target.data.long()).to(device)

            minibatch_unsup_size = configs['batch_size'] - configs['labeled_batch_size']
            minibatch_sup_size = configs['labeled_batch_size']

            # compute loss for unlabeled input
            [output_unsup, xhat_unsup, loss_pn_unsup, loss_bnmm_unsup] = net(input_unsup_var)
            loss_reconst_unsup = L2_loss(xhat_unsup, input_unsup_var).mean()
            softmax_unsup = F.softmax(output_unsup)
            loss_kl_unsup = -torch.sum(torch.log(10.0*softmax_unsup + 1e-8) * softmax_unsup) / minibatch_unsup_size
            loss_unsup = configs['alpha_reconst'] * loss_reconst_unsup + configs['alpha_kl'] * loss_kl_unsup + configs['alpha_bnmm'] * loss_bnmm_unsup + configs['alpha_pn'] * loss_pn_unsup



            # compute loss for labeled input
            [output_sup, xhat_sup, loss_pn_sup, loss_bnmm_sup] = net(input_sup_var, target_sup_var.squeeze_())
            loss_xentropy_sup = criterion(output_sup, target_sup_var) / minibatch_sup_size
            loss_reconst_sup = L2_loss(xhat_sup, input_sup_var).mean()
            softmax_sup = F.softmax(output_sup)
            loss_kl_sup = -torch.sum(torch.log(10.0*softmax_sup + 1e-8) * softmax_sup)/ minibatch_sup_size
            loss_sup = loss_xentropy_sup + configs['alpha_reconst'] * loss_reconst_sup + configs['alpha_kl'] * loss_kl_sup + configs['alpha_bnmm'] * loss_bnmm_sup + configs['alpha_pn'] * loss_pn_sup

            loss = torch.mean(loss_unsup + loss_sup)

            # compute the grads and update the parameters
            trainer.zero_grad()
            loss.backward()
            trainer.step()

            # accumulate all the losses for visualization
            loss_reconst = loss_reconst_unsup + loss_reconst_sup
            loss_pn = loss_pn_unsup + loss_pn_sup
            loss_xentropy = loss_xentropy_sup
            loss_kl = loss_kl_unsup + loss_kl_sup
            loss_bnmm = loss_bnmm_unsup + loss_bnmm_sup

            train_loss_xentropy += torch.mean(loss_xentropy).cpu().detach().numpy()
            train_loss_reconst += torch.mean(loss_reconst).cpu().detach().numpy()
            train_loss_pn += torch.mean(loss_pn).cpu().detach().numpy()
            train_loss_kl += torch.mean(loss_kl).cpu().detach().numpy()
            train_loss_bnmm += torch.mean(loss_bnmm).cpu().detach().numpy()
            train_loss += torch.mean(loss).cpu().detach().numpy()
            correct += get_acc(output_sup, target_sup_var).cpu().detach().numpy()

            num_batch_train += 1
            iter_indx += 1
        # writer.add_scalars('loss', {'train': train_loss / num_batch_train}, epoch)
        # writer.add_scalars('loss_xentropy', {'train': train_loss_xentropy / num_batch_train}, epoch)
        # writer.add_scalars('loss_reconst', {'train': train_loss_reconst / num_batch_train}, epoch)
        # writer.add_scalars('loss_pn', {'train': train_loss_pn / num_batch_train}, epoch)
        # writer.add_scalars('loss_kl', {'train': train_loss_kl / num_batch_train}, epoch)
        # writer.add_scalars('loss_bnmm', {'train': train_loss_bnmm / num_batch_train}, epoch)
        # writer.add_scalars('acc', {'train': correct / num_batch_train}, epoch)

 

        # Validation
        valid_loss = 0; valid_loss_xentropy = 0; valid_loss_reconst = 0; valid_loss_pn = 0; valid_loss_kl = 0; valid_loss_bnmm = 0
        valid_correct = 0
        num_batch_valid = 0
        valid_accuracy = 0
        valid_f1 = 0

        net.eval()

        for i, (batch, target) in enumerate(eval_loader):
            with torch.no_grad():
                input_var = torch.autograd.Variable(batch).to(device)
                target_var = torch.autograd.Variable(target.data.long()).to(device)

                minibatch_size = len(target_var)

                [output, xhat, loss_pn, loss_bnmm] = net(input_var, target_var)

                loss_xentropy = criterion(output, target_var.squeeze_())/minibatch_size
                loss_reconst = L2_loss(xhat, input_var).mean()
                softmax_val = F.softmax(output)
                loss_kl = -torch.sum(torch.log(10.0*softmax_val + 1e-8) * softmax_val)/minibatch_size
                loss = loss_xentropy + configs['alpha_reconst'] * loss_reconst + configs['alpha_kl'] * loss_kl + configs['alpha_bnmm'] * loss_bnmm + configs['alpha_pn'] * loss_pn

                valid_loss_xentropy += torch.mean(loss_xentropy).cpu().detach().numpy()
                valid_loss_reconst += torch.mean(loss_reconst).cpu().detach().numpy()
                valid_loss_pn += torch.mean(loss_pn).cpu().detach().numpy()
                valid_loss_kl += torch.mean(loss_kl).cpu().detach().numpy()
                valid_loss_bnmm += torch.mean(loss_bnmm).cpu().detach().numpy()
                valid_loss += torch.mean(loss).cpu().detach().numpy()
                valid_correct += get_acc(output, target_var).cpu().detach().numpy()
                
                accuracy, f1 = compute_metrics(target_var.cpu(), torch.argmax(output, dim=1, keepdim=False).cpu())
                valid_accuracy+=accuracy
                valid_f1+=f1
                num_batch_valid += 1
        
        valid_accuracies.append(valid_accuracy/num_batch_valid)
        f1_scores.append(valid_f1/num_batch_valid)
        valid_acc = valid_correct / num_batch_valid
        f1_s = valid_f1/num_batch_valid
        if f1_s > best_f1:
            best_f1 = f1_s
            torch.save(net.state_dict(), 'best_model.pth')
    #         torch.save(net.state_dict(), '%s/%s_best.pth'%(configs['model_dir, configs['exp_name))
    #     writer.add_scalars('loss', {'valid': valid_loss / num_batch_valid}, epoch)
    #     writer.add_scalars('loss_xentropy', {'valid': valid_loss_xentropy / num_batch_valid}, epoch)
    #     writer.add_scalars('loss_reconst', {'valid': valid_loss_reconst / num_batch_valid}, epoch)
    #     writer.add_scalars('loss_pn', {'valid': valid_loss_pn / num_batch_valid}, epoch)
    #     writer.add_scalars('loss_kl', {'valid': valid_loss_kl / num_batch_valid}, epoch)
    #     writer.add_scalars('loss_bnmm', {'valid': valid_loss_bnmm / num_batch_valid}, epoch)
    #     writer.add_scalars('acc', {'valid': valid_acc}, epoch)
        epoch_str = ("Epoch %d. Train Loss: %f, Train Xent: %f, Train Reconst: %f, Train Pn: %f, Train acc %f, Valid Loss: %f, Valid acc %f, Best f1 acc %f,f1 %f, acc %f "
                     % (epoch, train_loss / num_batch_train, train_loss_xentropy / num_batch_train, train_loss_reconst / num_batch_train, train_loss_pn / num_batch_train,
                        correct / num_batch_train, valid_loss / num_batch_valid, valid_acc, best_f1,valid_f1/num_batch_valid,valid_accuracy/num_batch_valid))
    #     if not epoch % 20:
    #         torch.save(net.state_dict(), '%s/%s_epoch_%i.pth'%(configs['model_dir, configs['exp_name, epoch))

    #     logging.info(epoch_str + time_str + ', lr ' + str(learning_rate))
        print(epoch_str)   
    #     return best_valid_acc
if __name__ == '__main__':
    train, labeled = load_datasets(
        "/rap/jvb-000-aa/COURS2019/etudiants/data/horoma", None, overlapped=False)
