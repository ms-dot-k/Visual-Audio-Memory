import argparse
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.models.visual_front import Visual_front
from src.models.audio_front import Audio_front
from src.models.classifier import GRU_Backend, Temp_classifier
from src.models.memory import Memory
import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.data.vid_aud_lrw import MultiDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
import torch.utils.data.distributed
import torch.nn.parallel
import math
from matplotlib import pyplot as plt
import time
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrw', default="DIR")
    parser.add_argument("--checkpoint_dir", type=str, default='./data/checkpoints/MultimodalMEM')
    parser.add_argument("--checkpoint", type=str, default='./data/GRU_Back_Ckpt.ckpt')
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--workers", type=int, default=3)

    parser.add_argument("--mode", type=str, default='train', help='train, test, val')

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--reduce_lr_at_1epoch", type=bool, default=True)
    parser.add_argument("--endurance", type=int, default=5, help='the lr will be halved if there is no improvement for "endurance" epochs')
    parser.add_argument("--opt", type=str, default='sgd')

    ##### Visual-Audio Memory (VAM) #####
    parser.add_argument("--radius", type=float, default=16.0)
    parser.add_argument("--n_slot", type=int, default=88)

    parser.add_argument("--backend", type=str, default='GRU', help='GRU, MSTCN')

    parser.add_argument("--max_timesteps", type=int, default=29)
    parser.add_argument("--augmentations", default=True)
    parser.add_argument("--test_aug", default=True)

    parser.add_argument("--dataparallel", default=True)
    parser.add_argument("--distributed", default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()
    return args


def train_net(args):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.local_rank)
    torch.cuda.manual_seed_all(args.local_rank)
    random.seed(args.local_rank)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_PORT'] = '5555'

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    train_data = MultiDataset(
        lrw=args.lrw,
        mode='train',
        max_v_timesteps=args.max_timesteps,
        augmentations=args.augmentations,
    )

    v_front = Visual_front(in_channels=1)
    a_front = Audio_front(in_channels=1)
    mem = Memory(radius=args.radius, n_slot=args.n_slot)

    assert args.backend in ['GRU', 'MSTCN'], 'Wrong Backend'
    if args.backend == 'GRU':
        back = GRU_Backend()
    else:
        back = Temp_classifier()

    params = [{'params': mem.parameters(), 'name': 'memory'},
              {'params': v_front.parameters(), 'name': 'v_front'},
              {'params': a_front.parameters(), 'name': 'a_front'},
              {'params': back.parameters(), 'name': 'backend'}
              ]

    if args.opt == 'sgd':
        f_optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        f_optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    f_scheduler = None

    if args.checkpoint is not None:
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        a_front.load_state_dict(checkpoint['a_front_state_dict'])
        v_front.load_state_dict(checkpoint['v_front_state_dict'])
        mem.load_state_dict(checkpoint['mem_state_dict'])
        back.load_state_dict(checkpoint['back_state_dict'])
        del checkpoint

    v_front.cuda()
    a_front.cuda()
    mem.cuda()
    back.cuda()

    if args.distributed:
        v_front = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v_front)
        a_front = torch.nn.SyncBatchNorm.convert_sync_batchnorm(a_front)
        mem = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mem)
        back = torch.nn.SyncBatchNorm.convert_sync_batchnorm(back)

    if args.distributed:
        v_front = DDP(v_front, device_ids=[args.local_rank], output_device=args.local_rank)
        a_front = DDP(a_front, device_ids=[args.local_rank], output_device=args.local_rank)
        mem = DDP(mem, device_ids=[args.local_rank], output_device=args.local_rank)
        back = DDP(back, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.dataparallel:
        v_front = DP(v_front)
        a_front = DP(a_front)
        mem = DP(mem)
        back = DP(back)

    if args.mode == 'train':
        train(v_front, a_front, mem, back, train_data, args.epochs, optimizer=f_optimizer, scheduler=f_scheduler, args=args)
    elif args.mode == 'val':
        validate(v_front, mem, back, fast_validate=True)
    else:
        test(v_front, mem, back)

def train(v_front, a_front, mem, back, train_data, epochs, optimizer, scheduler, args):
    best_val_acc = 0.0
    endurance = 0
    if args.local_rank == 0:
        writer = SummaryWriter(comment=os.path.split(args.checkpoint_dir)[-1])

    v_front.train()
    a_front.train()
    mem.train()
    back.train()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    dataloader = DataLoader(
        train_data,
        shuffle=False if args.distributed else True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    criterion = nn.CrossEntropyLoss().cuda()

    samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    step = 0
    for epoch in range(args.start_epoch, epochs):
        loss_list = []
        acc_list = []
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print(f"Epoch [{epoch}/{epochs}]")
        prev_time = time.time()
        for i, batch in enumerate(dataloader):
            step += 1
            if args.local_rank == 0 and i % 100 == 0:
                iter_time = (time.time() - prev_time) / 100
                prev_time = time.time()
                print("******** Training [%d / %d] : %d / %d, Iter Time : %.3f sec, Learning Rate of %s: %f ********" % (
                epoch, epochs, (i + 1) * batch_size, samples, iter_time, optimizer.param_groups[0]['name'], optimizer.param_groups[0]['lr']))
            a_in, v_in, target = batch

            v_feat = v_front(v_in.cuda())   #B,S,512
            a_feat = a_front(a_in.cuda())   #B,S,512

            te_fusion, tr_fusion, recon_loss, add_loss = mem(v_feat, a_feat, inference=False)

            te_mem_pred = back(te_fusion)
            tr_mem_pred = back(tr_fusion)

            recon_loss = recon_loss.mean(0)
            add_loss = add_loss.mean(0)

            te_mem_loss = criterion(te_mem_pred, target.long().cuda())
            tr_mem_loss = criterion(tr_mem_pred, target.long().cuda())

            tot_loss = te_mem_loss + tr_mem_loss + add_loss + recon_loss

            te_m_prediction = torch.argmax(te_mem_pred, dim=1).cpu().numpy()
            te_m_acc = np.mean(te_m_prediction == target.long().numpy())
            tr_m_prediction = torch.argmax(tr_mem_pred, dim=1).cpu().numpy()
            tr_m_acc = np.mean(tr_m_prediction == target.long().numpy())

            loss_list.append(te_mem_loss.cpu().item())
            acc_list.append(te_m_acc)

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

            if args.distributed:
                dist.barrier()

            if args.local_rank == 0:
                if i % 100 == 0:
                    for j in range(2):
                        print('label: ', train_data.word_list[target.long().numpy()[j]])
                        print('mem_prediction: ', train_data.word_list[te_m_prediction[j]])
                if writer is not None:
                    writer.add_scalar('Mem/te_loss', te_mem_loss.cpu().item(), step)
                    writer.add_scalar('Mem/te_acc', te_m_acc, step)
                    writer.add_scalar('Mem/tr_loss', tr_mem_loss.cpu().item(), step)
                    writer.add_scalar('Mem/tr_acc', tr_m_acc, step)
                    writer.add_scalar('Mem/add_loss', add_loss.cpu().item(), step)
                    writer.add_scalar('Mem/recon_loss', recon_loss.cpu().item(), step)

                    for o in range(len(optimizer.param_groups)):
                        writer.add_scalar('lr/%s_lr' % optimizer.param_groups[o]['name'], optimizer.param_groups[o]['lr'], step)

        if args.local_rank == 0:
            logs = validate(v_front, mem, back, epoch=epoch, writer=writer)
        else:
            logs = validate(v_front, mem, back, epoch=epoch, writer=None)

        if args.local_rank == 0:
            print('VAL_ACC: ', logs[1])
            print('Saving checkpoint: %d' % epoch)
            if args.distributed or args.dataparallel:
                v_state_dict = v_front.module.state_dict()
                a_state_dict = a_front.module.state_dict()
                mem_state_dict = mem.module.state_dict()
                back_state_dict = back.module.state_dict()
            else:
                v_state_dict = v_front.state_dict()
                a_state_dict = a_front.state_dict()
                mem_state_dict = mem.state_dict()
                back_state_dict = back.state_dict()
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            torch.save({'v_front_state_dict': v_state_dict, 'a_front_state_dict': a_state_dict, 'mem_state_dict': mem_state_dict, 'back_state_dict': back_state_dict},
                       os.path.join(args.checkpoint_dir, 'Epoch_%04d_acc_%.5f.ckpt' % (epoch, logs[1])))

        if logs[1] > best_val_acc:
            endurance = 0
            best_val_acc = logs[1]
            if args.local_rank == 0:
                bests = glob.glob(os.path.join(args.checkpoint_dir, 'Best_*.ckpt'))
                for prev in bests:
                    os.remove(prev)
                torch.save({'v_front_state_dict': v_state_dict, 'a_front_state_dict': a_state_dict, 'mem_state_dict': mem_state_dict, 'back_state_dict': back_state_dict},
                           os.path.join(args.checkpoint_dir, 'Best_%04d_acc_%.5f.ckpt' % (epoch, logs[1])))
        else:
            endurance += 1

        if epoch == 1 and args.reduce_lr_at_1epoch:
            endurance = 0
            print('################# Reduce First LR #################')
            for g in optimizer.param_groups:
                g['lr'] = 0.01

        elif endurance >= args.endurance:
            endurance = 0
            print('################# Reduce LR #################')
            for g in optimizer.param_groups:
                current_lr = g['lr']
                g['lr'] = current_lr * 0.50

        if scheduler is not None:
            scheduler.step()

    if args.local_rank == 0:
        print('Finishing training')


def validate(v_front, mem, back, fast_validate=False, epoch=0, writer=None):
    with torch.no_grad():
        v_front.eval()
        mem.eval()
        back.eval()

        val_data = MultiDataset(
            lrw=args.lrw,
            mode='val',
            max_v_timesteps=args.max_timesteps,
            augmentations=False,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size * 2,
            num_workers=args.workers,
            drop_last=False
        )

        criterion = nn.CrossEntropyLoss().cuda()
        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(2 * batch_size, int(len(dataloader.dataset)))
            max_batches = 2
        else:
            samples = int(len(dataloader.dataset))
            max_batches = int(len(dataloader))

        val_loss = []
        tot_cor, tot_v_cor, tot_a_cor, tot_num = 0, 0, 0, 0

        description = 'For checking validation step' if fast_validate else 'Validation'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 10 == 0:
                if not fast_validate:
                    print("******** Validation : %d / %d ********" % ((i + 1) * batch_size, samples))
            _, v_in, target = batch

            v_feat = v_front(v_in.cuda())  # S,B,512
            te_fusion, _, _, _ = mem(v_feat, None, inference=True)
            te_m_pred = back(te_fusion)

            loss = criterion(te_m_pred, target.long().cuda()).cpu().item()
            prediction = torch.argmax(te_m_pred.cpu(), dim=1).numpy()

            tot_cor += np.sum(prediction == target.long().numpy())
            tot_num += len(prediction)

            batch_size = te_m_pred.size(0)
            val_loss.append(loss)

            if i >= max_batches:
                break

        if args.local_rank == 0:
            if writer is not None:
                writer.add_scalar('Val/mem_loss', np.mean(np.array(val_loss)), epoch)
                writer.add_scalar('Val/mem_acc', tot_cor / tot_num, epoch)

        v_front.train()
        mem.train()
        back.train()
        print('Validation_ACC:', tot_cor / tot_num)
        if fast_validate:
            return {}
        else:
            return np.mean(np.array(val_loss)), tot_cor / tot_num

def test(v_front, mem, back):
    with torch.no_grad():
        v_front.eval()
        mem.eval()
        back.eval()

        val_data = MultiDataset(
            lrw=args.lrw,
            mode='test',
            max_v_timesteps=args.max_timesteps,
            augmentations=False,
        )

        dataloader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size * 2,
            num_workers=args.workers,
            drop_last=False
        )

        criterion = nn.CrossEntropyLoss().cuda()
        batch_size = dataloader.batch_size
        samples = int(len(dataloader.dataset))
        max_batches = int(len(dataloader))

        test_loss = []
        tot_cor, tot_v_cor, tot_a_cor, tot_num = 0, 0, 0, 0

        description = 'Test'
        if args.local_rank == 0:
            print(description)
        for i, batch in enumerate(dataloader):
            if args.local_rank == 0 and i % 10 == 0:
                print("******** Test : %d / %d ********" % ((i + 1) * batch_size, samples))
            _, v_in, target = batch

            v_feat = v_front(v_in.cuda())  # S,B,512
            te_fusion, _, _, _ = mem(v_feat, None, inference=True)
            te_m_pred = back(te_fusion)
            loss = criterion(te_m_pred, target.long().cuda()).cpu().item()

            if args.test_aug:
                #B,C,T,H,W
                ori_pred = te_m_pred.clone()
                v_in = v_in.flip(4).cuda()
                v_feat = v_front(v_in)
                te_fusion, _, _, _ = mem(v_feat, None, inference=True)
                te_m_pred = back(te_fusion)
                te_m_pred += ori_pred

            prediction = torch.argmax(te_m_pred.cpu(), dim=1).numpy()

            tot_cor += np.sum(prediction == target.long().numpy())
            tot_num += len(prediction)

            batch_size = te_m_pred.size(0)
            test_loss.append(loss)

            if i >= max_batches:
                break

        print('Test_ACC:', tot_cor / tot_num)
        return np.mean(np.array(test_loss)), tot_cor / tot_num


if __name__ == "__main__":
    args = parse_args()
    train_net(args)

