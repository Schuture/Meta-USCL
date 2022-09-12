import os
import sys
import time
import math
import random
import argparse
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from model import ContrastiveModel, WNet
from dataloader import get_dataloaders
from loss import InfoNCE
from meta import MetaSGD


def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)  # disable hash randomness
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\nRunning on:', device)
    
    if device == 'cuda':
        device_name = torch.cuda.get_device_name()
        print('The device name is:', device_name)
        cap = torch.cuda.get_device_capability(device=None)
        print('The capability of this device is:', cap, '\n')
    
    return device


def get_lr(epoch):
    '''
    Get the cosine decreasing learning rate with exponential warmup.
    '''
    if epoch <= args.warmup: # from 0.1 to 1
        lr_ratio = 0.1 + 0.9 * math.exp(-5 * (1 - min(epoch / args.warmup, 1)) ** 2)
    else: # from 1 to 0
        lr_ratio = 0.5 * (1 + math.cos(math.pi*(epoch - args.warmup)/(args.epochs - args.warmup)))
        
    return lr_ratio


def get_model(device):
    '''
    Remove the last fc layer of a backbone and add an MLP projector.
    '''
    model = ContrastiveModel(model_name=args.model,
                             depth=args.depth,
                             feat_dim=args.feat_dim,
                             pretrained=args.pretrained)

    return model.to(device)


def save_ckpt(epoch, model, cmwnet, optimizer,
              optimizer_cmwnet, model_checkpoints_folder):
    '''
    Save the model parameters of main model and corresponding weight module.
    '''
    # 1. make directory
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    # 2. save model checkpoint
    model_ckpt = {'epoch': epoch,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(model_ckpt,
               os.path.join(model_checkpoints_folder, 'best_model.pth'))
    # 3. save cmwnet checkpoint
    if cmwnet:
        cmwnet_ckpt = {'state_dict': cmwnet.state_dict(),
                     'optimizer': optimizer_cmwnet.state_dict()}
        torch.save(cmwnet_ckpt,
                   os.path.join(model_checkpoints_folder, 'best_cmwnet.pth'))


def save_info(info, model_checkpoints_folder):
    '''
    Save the useful information generated during training.
    '''
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'info.txt'), 'a') as f:
        if info.startswith('Training'):
            info = '\n' + info
        if info.startswith('Model'):
            f.write(info)
        else:
            f.write('\n'+info)


def train_step(xl, xr, model, meta_model, cmwnet, 
               meta_loader, optimizer, optimizer_cmwnet, 
               criterion, apex_support, device):
    '''
    Update the main model for one step.
    
    If this process is done without meta learning, we forward pass the data
    and get the gradient to update the model directly. If this process is
    done with meta learning, we firstly use meta model to calculate its θ_{t+1}
    and use updated meta model to get the loss and gradient for weight module
    cmwnet. And then we use the updated cmwnet to calculate sample weights and
    update the main model with weighted loss.
    
    Time consumption: RTX 3060, worker=0, bs=32, image=112x112, 6.58s for an iteration
        1. first forward pass and update meta_model, 0.044s
        2. compute second order grad and update cmwnet, 6.2s
        3. forward pass and update model, 0.097s
        4. other operations take 0.25s (load data, and other codes)
    '''
    # Update weight net.
    if args.meta:
        meta_model.load_state_dict(model.state_dict())  # load the same parameters as model

        # 1. Forward pass for meta model.
        rl, zl = meta_model(xl)
        rr, zr = meta_model(xr)
        weights = cmwnet(torch.cat((rl, rr), 1))
        loss_hat = criterion(zl, zr, weights=weights)
        
        # 2. Update the meta model to get θ_{t+1}.
        meta_model.zero_grad()
        grads = torch.autograd.grad(loss_hat, (meta_model.parameters()),
                                    create_graph=True, allow_unused=True)
        pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=args.lr_meta)
        pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=args.lr_meta)
        pseudo_optimizer.load_state_dict(optimizer.state_dict())
        pseudo_optimizer.meta_step(grads)

        del grads

        # 3. Compute upper level objective.
        try:
            meta_xl, meta_xr = next(meta_iterator)
        except:
            meta_iterator = iter(meta_loader)
            meta_xl, meta_xr = next(meta_iterator)
        meta_xl = meta_xl.to(device)
        meta_xr = meta_xr.to(device)
        meta_rl, meta_zl = meta_model(meta_xl)
        meta_rr, meta_zr = meta_model(meta_xr)
        loss_meta = criterion(meta_zl, meta_zr, weights=None)

        # 4. Use meta loss to update weight net parameters α
        # with second order gradients. Very time consuming
        start = time.time()
        optimizer_cmwnet.zero_grad()
        if apex_support and args.fp16_precision:
            with amp.scale_loss(loss_meta, optimizer_cmwnet) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_meta.backward()
        optimizer_cmwnet.step()
        print('Time for second order optimization: ', round(time.time() - start, 2))

    # 5. Get the representations and the projections.
    rl, zl = model(xl)
    rr, zr = model(xr)
    with torch.no_grad():
        weights = cmwnet(torch.cat((rl, rr), 1)) if args.meta else None
    loss = criterion(zl, zr, weights=weights)
    
    # 6. Backward to get the gradient for main model.
    optimizer.zero_grad()
    if apex_support and args.fp16_precision:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()
    
    return loss


def validate(model, device, valid_loader, criterion):
    '''
    Validate the main model and get validation loss for this epoch.
    '''
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        counter = 0
        for data in valid_loader:
            xl, xr = data
            xl = xl.to(device)
            xr = xr.to(device)
            
            rl, zl = model(xl)
            rr, zr = model(xr)

            loss = criterion(zl, zr, weights=None)
            
            valid_loss += loss.item()
            counter += 1
        valid_loss /= counter
    model.train()
    
    return valid_loss


def train(model, meta_model, cmwnet, train_loader, valid_loader, meta_loader,
          optimizer, optimizer_cmwnet, criterion, device, model_checkpoints_folder):
    start_epoch = 1
    # load checkpoints
    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Model checkpoint from {} loaded'.format(args.ckpt))
        if cmwnet is not None:
            ckpt = args.ckpt.replace('model','cmwnet')
            checkpoint = torch.load(ckpt)
            cmwnet.load_state_dict(checkpoint['state_dict'])
            optimizer_cmwnet.load_state_dict(checkpoint['optimizer'])
            print('Weight net checkpoint from {} loaded'.format(ckpt))

    # Cosine learning rate with warmup
    lambda1 = lambda epoch: get_lr(epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    if apex_support and args.fp16_precision:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O2',
                                          keep_batchnorm_fp32=True)
        if cmwnet:
            cmwnet, optimizer_cmwnet = amp.initialize(cmwnet, optimizer_cmwnet,
                                                  opt_level='O2',
                                                  keep_batchnorm_fp32=True)
    
    start_time = time.time()
    epoch_start_time = time.time()
    epoch_end_time = time.time()
    best_epoch = 0
    best_valid_loss = np.inf
    for epoch in range(start_epoch, args.epochs + 1):
        for i, data in enumerate(train_loader, 1):
            try:
                # forward
                xl, xr = data  # N samples for left branch, N samples for right branch
                xl = xl.to(device)
                xr = xr.to(device)
                
                model.train()
                
                loss = train_step(xl, xr, model, meta_model, cmwnet, 
                                  meta_loader, optimizer, optimizer_cmwnet, 
                                  criterion, apex_support, device)
                # Meta model must be rebuild to make a new computational graph.
                if args.meta:
                    meta_model = get_model(device)
                
                epoch_start_time, epoch_end_time = epoch_end_time, time.time()
                training_info = 'Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Time: {:.2f}s'.format(
                    epoch, args.epochs, i, len(train_loader), loss, epoch_end_time - epoch_start_time)
                print(training_info)
                del loss
                if i == 1:
                    save_info(training_info, model_checkpoints_folder)
                time.sleep(0.001)
            except:
                print('miss error, skip this training iteration')
                traceback.print_exc()
        # validate the model
        if epoch % args.eval_every_n_epochs == 0 or epoch >= args.epochs - 30:
            try:
                epoch_start_time = time.time()
                valid_loss = validate(model, device, valid_loader, criterion)
                epoch_end_time = time.time()
                if valid_loss < best_valid_loss:
                    # save the model parameters
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    save_ckpt(epoch, model, cmwnet, optimizer,
                              optimizer_cmwnet, model_checkpoints_folder)
                
                valid_info = 'Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Time: {:.2f}s'.format(
                    epoch, args.epochs, len(valid_loader), len(valid_loader), valid_loss, epoch_end_time - epoch_start_time)
                print(valid_info)
                save_info(valid_info, model_checkpoints_folder)
            except:
                print('miss error, skip validation')
                traceback.print_exc()

        print('Learning rate this epoch:', scheduler.get_last_lr(), '\n')
        scheduler.step()
    
    total_time = time.time() - start_time
    finish_info = '\nTotal trining time: {}s'.format(round(total_time)) + \
        '\nThe min validation loss is: {}, reached at epoch {}.\n'.format(best_valid_loss, best_epoch)
    print(finish_info)
    save_info(finish_info, model_checkpoints_folder)


def main():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta USCL')

    # model
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', default='ResNet')
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--fp16_precision', action='store_true', default=False)
    # Whether we need ImageNet pretrained parameters to initialize the learning.
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--name', default=None)  # name of this training
    
    # optimization
    parser.add_argument('--optim', default='SGD')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--meta_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--meta', action='store_false', default=True,
                        help='whether to use meta learning')
    parser.add_argument('--lr_meta', type=float, default=1e-1)
    # Learning rate for weight net need to be carefully tuned.
    parser.add_argument('--lr_cmwnet', type=float, default=6e-5)
    parser.add_argument('--eval_every_n_epochs', type=int, default=1)
    
    # dataset
    parser.add_argument('--dataset', default='US-4')
    parser.add_argument('--data_path', default='Butterfly')
    # Multi worker is too burdensome for meta learning if your memory is limited.
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--valid_ratio', type=float, default=0.20)
    parser.add_argument('--input_shape', type=int, default=64) # large image sizes cause huge memory footprint
    parser.add_argument('--mixup', default='standard')
    parser.add_argument('--max_dist', type=int, default=999)
    parser.add_argument('--samples', type=int, default=3,
                        help='using how many frames in a video to generate a positive pair, 1-3')
    parser.add_argument('--aug_first', action='store_true', default=False)
    parser.add_argument('--cropping_size', type=float, default=0.7)
    parser.add_argument('--color_jitter', type=float, default=0.8)
    
    # loss
    parser.add_argument('--loss', default='InfoNCE')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # resume
    parser.add_argument('--ckpt', default=None)  # load ckpt

    args = parser.parse_args()
    
    arg_info = 'Model arguments:' + \
               '\n\tseed: ' + str(args.seed) + \
               '\n\tmodel: ' + str(args.model) + \
               '\n\tdepth: ' + str(args.depth) + \
               '\n\tfeature dimension: ' + str(args.feat_dim) + \
               '\n\tfp16: '+str(args.fp16_precision) + \
               '\n\tImageNet pretrained: ' + str(args.pretrained) + \
               '\n\tname: '+str(args.name) + \
               '\nOptimization arguments:' + \
               '\n\toptimizer: ' + str(args.optim) + \
               '\n\tepochs: ' + str(args.epochs) + \
               '\n\tbatch size: ' + str(args.batch_size) + \
               '\n\ttest batch size: ' + str(args.valid_batch_size) + \
               '\n\tmeta batch size: ' + str(args.meta_batch_size) + \
               '\n\twarm-up epochs: ' + str(args.warmup) + \
               '\n\tlearning rate: ' + str(args.lr) + \
               '\n\tmeta learning: ' + str(args.meta) + \
               '\n\tmeta learning rate: ' + str(args.lr_meta) + \
               '\n\tweight net learning rate: ' + str(args.lr_cmwnet) + \
               '\nDataset arguments:' + \
               '\n\tdataset: ' + str(args.dataset) + \
               '\n\tdata path: ' + str(args.data_path) + \
               '\n\tworkers: ' + str(args.workers) + \
               '\n\tvalid ratio: ' + str(args.valid_ratio) + \
               '\n\tinput shape: ' + str(args.input_shape) + \
               '\n\tmixup: ' + str(args.mixup) + \
               '\n\tmax distance: ' + str(args.max_dist) + \
               '\n\tsamples: ' + str(args.samples) + \
               '\n\taugment before mixup: ' + str(args.aug_first) + \
               '\n\tcropping size: ' + str(args.cropping_size) + \
               '\n\tcolor jitter: ' + str(args.color_jitter) + \
               '\nLoss arguments:' + \
               '\n\tloss: ' + str(args.loss) + \
               '\n\ttemperature: ' + str(args.temperature) + \
               '\n\tweight decay: ' + str(args.weight_decay) + \
               '\nResume arguments:' + \
               '\n\tckpt: '+str(args.ckpt)
    
    # Initialize the ckpt/info folder.
    if args.name is not None:
        model_checkpoints_folder = os.path.join('checkpoint', 'checkpoint_' + args.name)
    else:
        model_checkpoints_folder = os.path.join('checkpoint', 
                'checkpoint_' + '_'.join(time.asctime(time.localtime(time.time())).split()))
    model_checkpoints_folder = model_checkpoints_folder.replace(':', '_')
    print('model_checkpoint_folder:', model_checkpoints_folder)
    
    print(arg_info)
    save_info(arg_info, model_checkpoints_folder)
    
    # Set seed.
    if isinstance(args.seed, int):
        seed_torch(seed=args.seed)
    
    apex_support = False
    try:
        sys.path.append('./apex')
        from apex import amp
        print('Apex on, the programe can run on mixed precision.')
        apex_support = True
    except:
        print('Please install apex for mixed precision training '
              'from: https://github.com/NVIDIA/apex')
        apex_support = False

    device = get_device()
    
    data_loaders = get_dataloaders(dataset=args.dataset, 
                                   data_path=args.data_path,
                                   meta=args.meta,                                   
                                   batch_size=args.batch_size,
                                   valid_batch_size=args.valid_batch_size,
                                   meta_batch_size=args.meta_batch_size,
                                   workers=args.workers,
                                   valid_ratio=args.valid_ratio,
                                   mixup=args.mixup,
                                   max_dist=args.max_dist,
                                   samples=args.samples,
                                   input_shape=args.input_shape,
                                   aug_first=args.aug_first,
                                   cropping_size=args.cropping_size,
                                   color_jitter=args.color_jitter)
    train_loader, valid_loader, meta_loader = data_loaders

    print('\nmodel:')
    model = get_model(device)  # initialize a backbone and add an MLP projector
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD( # Optimizer must be SGD for Meta-SGD
            model.parameters(), # should be model.parameters() for wideresnet
            lr=args.lr, # 0.1 for SGD, Adam can be much smaller
            momentum=0.9,
            dampening=0,
            weight_decay=args.weight_decay,
            nesterov=False,
        )
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, weight_decay=args.weight_decay)
    
    if args.meta:
        print('meta model:')
        meta_model = get_model(device)
        num_ftrs = model.num_ftrs
        cmwnet = WNet(2*num_ftrs, 10, 1).to(device)
        optimizer_cmwnet = torch.optim.Adam(cmwnet.parameters(), lr=args.lr_cmwnet)
    else:
        meta_model = None
        cmwnet = None
        optimizer_cmwnet = None

    if args.loss == 'InfoNCE':
        criterion = InfoNCE(device, args.batch_size,
                            args.temperature, use_cosine_similarity=True)
    else:
        raise ValueError('Only InfoNCE loss is supported for now.')
        
    train(model, meta_model, cmwnet, train_loader, valid_loader, meta_loader,
          optimizer, optimizer_cmwnet, criterion, device, model_checkpoints_folder)





