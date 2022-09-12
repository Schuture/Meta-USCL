import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from matplotlib import pyplot as plt

from my_dataset import COVIDDataset

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    # ============================ step 1/5 Data ============================

    train_transform = transforms.Compose([
        transforms.Resize((args.input_shape, args.input_shape)),
        transforms.RandomResizedCrop(size=args.input_shape, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((args.input_shape, args.input_shape)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])

    train_data = COVIDDataset(data_dir=data_dir, train=True, transform=train_transform)
    valid_data = COVIDDataset(data_dir=data_dir, train=False, transform=valid_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=args.workers)

    # ============================ step 2/5 Model ============================
    if args.model == 'ResNet':
        if args.depth == 18:
            net = models.resnet18(pretrained=args.imagenet)
        elif args.depth == 34:
            net = models.resnet34(pretrained=args.imagenet)
        elif args.depth == 50:
            net = models.resnet50(pretrained=args.imagenet)
    elif args.model == 'ShuffleNet':
        if args.depth == 1:
            net = models.shufflenet_v2_x0_5(pretrained=args.imagenet)
        elif args.depth == 2:
            net = models.shufflenet_v2_x1_0(pretrained=args.imagenet)
        elif args.depth == 3:
            net = models.shufflenet_v2_x1_5(pretrained=args.imagenet)
        elif args.depth == 4:
            net = models.shufflenet_v2_x2_0(pretrained=args.imagenet)

    if args.supervise == 'self' and not args.imagenet:
        state_dict = torch.load(state_dict_path)['state_dict']
        state_dict = {k:state_dict[k] for k in list(state_dict.keys()) if not (k.startswith('l') or k.startswith('fc'))}
        state_dict = {k:state_dict[k] for k in list(state_dict.keys()) if not k.startswith('classifier')}
        
        con_layer_names = list(state_dict.keys())
        target_layer_names = list(net.state_dict().keys())
        new_dict = {target_layer_names[i]:state_dict[con_layer_names[i]] for i in range(len(con_layer_names))}

        model_dict = net.state_dict()
        model_dict.update(new_dict)
        net.load_state_dict(model_dict)
        print('\nThe self-supervised trained parameters are loaded.\n')

    for name, param in net.named_parameters():
        if finetune_mode == 'all':
            break
        param.requires_grad = False
        if finetune_mode == 'last_three':
            if args.model == 'ResNet' and args.depth == 18 and name.startswith('layer4.1'):
                param.requires_grad = True
            if args.model == 'ResNet' and args.depth == 34 and name.startswith('layer4.2'):
                param.requires_grad = True
            if args.model == 'ResNet' and args.depth == 50 and name.startswith('layer4.2'):
                param.requires_grad = True
            if args.model == 'ShuffleNet' and name.startswith('conv5'):
                param.requires_grad = True

    # Replace the fc layers
    if args.model == 'ResNet':
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 3)
    elif args.model == 'DenseNet':
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Linear(num_ftrs, 3)
    elif args.model == 'ShuffleNet':
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 3)
    elif args.model == 'WideResNet':
        net.fc = nn.Linear(net.nChannels, 3)

    for name, param in net.named_parameters():
        print(name, '\t', 'requires_grad=', param.requires_grad)

    net.to(device)

    # ============================ step 3/5 Loss ============================
    criterion = nn.CrossEntropyLoss()

    # ============================ step 4/5 Optimizer ============================
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=MAX_EPOCH, 
                                                     eta_min=0,
                                                     last_epoch=-1)

    # ============================ step 5/5 Training ============================
    print('\nTraining start!\n')
    start = time.time()
    train_curve = list()
    valid_curve = list()
    max_acc = 0.
    reached = 0 # which epoch reached the max accuracy

    # the statistics of classification result: classification_results[true][pred]
    classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    best_classification_results = None

    if apex_support and fp16_precision:
        net, optimizer = amp.initialize(net, optimizer,
                                        opt_level='O2',
                                        keep_batchnorm_fp32=True)
    for epoch in range(1, MAX_EPOCH + 1):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # Forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            # Backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            if apex_support and fp16_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()

            # Training info
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        print('Learning rate this epoch:', scheduler.get_lr()[0])
        scheduler.step()  # Update LR

        # validate the model
        if epoch % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).cpu().squeeze().sum().numpy()
                    for k in range(len(predicted)):
                        classification_results[labels[k]][predicted[k]] += 1 # "label" is regarded as "predicted"

                    loss_val += loss.item()

                acc = correct_val / total_val
                if acc > max_acc:
                    max_acc = acc
                    reached = epoch
                    best_classification_results = classification_results
                classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                valid_curve.append(loss_val/valid_loader.__len__())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, acc))

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(MAX_EPOCH, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))

    print('\nThe best prediction results of the dataset:')
    print('Class 0 predicted as class 0:', best_classification_results[0][0])
    print('Class 0 predicted as class 1:', best_classification_results[0][1])
    print('Class 0 predicted as class 2:', best_classification_results[0][2])
    print('Class 1 predicted as class 0:', best_classification_results[1][0])
    print('Class 1 predicted as class 1:', best_classification_results[1][1])
    print('Class 1 predicted as class 2:', best_classification_results[1][2])
    print('Class 2 predicted as class 0:', best_classification_results[2][0])
    print('Class 2 predicted as class 1:', best_classification_results[2][1])
    print('Class 2 predicted as class 2:', best_classification_results[2][2])

    try:
        acc0 = best_classification_results[0][0] / sum(best_classification_results[i][0] for i in range(3))
        recall0 = best_classification_results[0][0] / sum(best_classification_results[0])
        print('\nClass 0 accuracy:', acc0)
        print('Class 0 recall:', recall0)
        print('Class 0 F1:', 2 * acc0 * recall0 / (acc0 + recall0))

        acc1 = best_classification_results[1][1] / sum(best_classification_results[i][1] for i in range(3))
        recall1 = best_classification_results[1][1] / sum(best_classification_results[1])
        print('\nClass 1 accuracy:', acc1)
        print('Class 1 recall:', recall1)
        print('Class 1 F1:', 2 * acc1 * recall1 / (acc1 + recall1))

        acc2 = best_classification_results[2][2] / sum(best_classification_results[i][2] for i in range(3))
        recall2 = best_classification_results[2][2] / sum(best_classification_results[2])
        print('\nClass 2 accuracy:', acc2)
        print('Class 2 recall:', recall2)
        print('Class 2 F1:', 2 * acc2 * recall2 / (acc2 + recall2))
    except:
        print('meet 0 denominator')
    
    return best_classification_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear evaluation')
    parser.add_argument('-m', '--model', default='ResNet')
    parser.add_argument('-d', '--depth', type=int, default=18)
    parser.add_argument('--input_shape', type=int, default=64)
    parser.add_argument('-p', '--path', default='resnet18_64.pth', help='path of ckpt')
    parser.add_argument('--data_dir', default='covid_5_fold', help='path dataset')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-w', '--workers', type=int, default=0)
    parser.add_argument('-S', '--supervise', default='self') # semi=ResNet，self=ResNetSimCLR
    parser.add_argument('-i', '--imagenet', action='store_true', default=False)
    args = parser.parse_args()

    apex_support = False
    try:
        sys.path.append('./apex')
        from apex import amp
        print("Apex on, run on mixed precision.")
        apex_support = True
    except:
        print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
        apex_support = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nRunning on:", device)

    if device == 'cuda':
        device_name = torch.cuda.get_device_name()
        print("The device name is:", device_name)
        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')
    
    # Hyper-parameters
    MAX_EPOCH = 30
    BATCH_SIZE = 128
    LR = 0.01
    weight_decay = 1e-4
    log_interval = 2
    val_interval = 1
    base_path = ""
    state_dict_path = os.path.join(base_path, args.path)
    print('State dict path:', state_dict_path)
    fp16_precision = True
    finetune_mode = 'last_three' # 'last_one' / 'last_three' / 'all'
    
    print('\n=================Evaluation on POCUS dataset start===================')
    results = {}
    
    print('\n\n*************************************************************************')
    print('\nThe finetune mode is', finetune_mode, '\n')
    set_seed(args.seed)
    print('random seed:', args.seed)
    confusion_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(1, 6):
        print('\n' + '='*20 + 'The training of fold {} start.'.format(i) + '='*20)
        data_dir = args.data_dir + "/covid_data{}.pkl".format(i)
        best_classification_results = main()
        confusion_matrix = confusion_matrix + np.array(best_classification_results)

    print('\nThe confusion matrix is:')
    print(confusion_matrix)
    print('\nThe precision of class 0 is:', confusion_matrix[0,0] / sum(confusion_matrix[:,0]))
    print('The precision of class 1 is:', confusion_matrix[1,1] / sum(confusion_matrix[:,1]))
    print('The precision of class 2 is:', confusion_matrix[2,2] / sum(confusion_matrix[:,2]))
    print('\nThe recall of class 0 is:', confusion_matrix[0,0] / sum(confusion_matrix[0]))
    print('The recall of class 1 is:', confusion_matrix[1,1] / sum(confusion_matrix[1]))
    print('The recall of class 2 is:', confusion_matrix[2,2] / sum(confusion_matrix[2]))
    
    acc = (confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2])/confusion_matrix.sum()
    acc = round(acc, 4) * 100
    print('\nTotal acc is:', acc)

    # write logs
    with open(os.path.join(state_dict_path[:-14], 'finetune_POCUS.txt'), 'a') as f:
        f.write("The finetune model is " + str(args.model) + str(args.depth) + '\n')
        f.write("ImageNet:  " + str(args.imagenet) + '\n')
        f.write("Supervise:  " + str(args.supervise) + '\n')
        f.write("The finetune mode is " + finetune_mode + '\n')
        f.write('random seed: ' + str(args.seed))
        f.write('\nThe confusion matrix is:\n' + str(confusion_matrix))
        f.write('\nThe precision of class 0 is: ' + str(confusion_matrix[0,0] / sum(confusion_matrix[:,0])))
        f.write('\nThe precision of class 1 is: ' + str(confusion_matrix[1,1] / sum(confusion_matrix[:,1])))
        f.write('\nThe precision of class 2 is: ' + str(confusion_matrix[2,2] / sum(confusion_matrix[:,2])))
        f.write('\nThe recall of class 0 is: ' + str(confusion_matrix[0,0] / sum(confusion_matrix[0])))
        f.write('\nThe recall of class 1 is: ' + str(confusion_matrix[1,1] / sum(confusion_matrix[1])))
        f.write('\nThe recall of class 2 is: ' + str(confusion_matrix[2,2] / sum(confusion_matrix[2])))
        f.write('\nTotal acc is: ' + str(acc) + '\n\n\n')









