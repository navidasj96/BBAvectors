import torch
import torch.nn as nn
import os
import numpy as np
import loss
import cv2
import func_utils

from tqdm import tqdm

import argparse
import eval
from datasets.dataset_dota import DOTA
from datasets.dataset_hrsc import HRSC
from models import ctrbox_net
import decoder
import os

from datasets.dotadevkit.dotadevkit.evaluate import task1

def parse_args():
        parser = argparse.ArgumentParser(description='BBAVectors Implementation')
        parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
        parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
        parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Initial learning rate')
        parser.add_argument('--input_h', type=int, default=800, help='Resized image height')
        parser.add_argument('--input_w', type=int, default=800, help='Resized image width')
        parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
        parser.add_argument('--conf_thresh', type=float, default=0.18, help='Confidence threshold, 0.1 for general evaluation')
        parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
        parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
        parser.add_argument('--resume', type=str, default='model_last.pth', help='Weights resumed in testing and evaluation')
        parser.add_argument('--dataset', type=str, default='dota', help='Name of dataset')
        parser.add_argument('--data_dir', type=str, default='../Datasets/dota', help='Data directory')
        parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, eval}')
        parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
        args = parser.parse_args()
        return args
def evaluate():
  args = parse_args() 
  dataset = {'dota': DOTA, 'hrsc': HRSC}                  
  num_classes = {'dota': 2, 'hrsc': 1}
  heads = {'hm': num_classes[args.dataset],
    'wh': 10,
    'reg': 2,
    'cls_theta': 1
    }
  down_ratio = 4
  model = ctrbox_net.CTRBOX(heads=heads,
                            pretrained=True,
                            down_ratio=down_ratio,
                            final_kernel=1,
                            head_conv=256)

  # decoder = decoder.DecDecoder(K=args.K,
  #                             conf_thresh=args.conf_thresh,
  #                             num_classes=num_classes[args.dataset])                    
  ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder.DecDecoder(K=args.K,
                              conf_thresh=args.conf_thresh,
                              num_classes=num_classes[args.dataset]))
  ctrbox_obj.evaluation(args, down_ratio=down_ratio)
  task1.evaluate_result()


  

def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio):
        torch.manual_seed(317)
        self.dataset = dataset
        self.dataset_phase = {'dota': ['train'],
                              'hrsc': ['train', 'test']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio

    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    

    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer, epoch

    def train_network(self, args):

        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        save_path = 'weights_'+args.dataset
        start_epoch = 1
        
        # add resume part for continuing training when break previously, 10-16-2020
        if args.resume_train:
            self.model, self.optimizer, start_epoch = self.load_model(self.model, 
                                                                        self.optimizer, 
                                                                        args.resume_train, 
                                                                        strict=True)
            start_epoch+=1
        # end 

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if args.ngpus>1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        criterion = loss.LossAll()
        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio)
                 for x in self.dataset_phase[args.dataset]}

        dsets_loader = {}
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=True,
                                                           collate_fn=collater)

        print('Starting training...')
        train_loss = []
        ap_list = []
        for epoch in range(start_epoch, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)
            train_loss.append(epoch_loss)
            self.scheduler.step(epoch)

            np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_loss, fmt='%.6f')

            if epoch % 5 == 0 or epoch >= 1:
                self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer)

                self.save_model(os.path.join('/content/drive/MyDrive/VAID-OBB/OneClassWeightsAugmented', 'model_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer)
                              

                self.save_model(os.path.join(save_path, 'model_last.pth'),
                            epoch,
                            self.model,
                            self.optimizer)

            evaluate()
            if 'test' in self.dataset_phase[args.dataset] and epoch%5==0:
                mAP = self.dec_eval(args, dsets['test'])
                ap_list.append(mAP)
                np.savetxt(os.path.join(save_path, 'ap_list.txt'), ap_list, fmt='%.6f')

            self.save_model(os.path.join(save_path, 'model_last.pth'),
                            epoch,
                            self.model,
                            self.optimizer)

    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        for data_dict in tqdm(data_loader):
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)

            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss


    def dec_eval(self, args, dsets):
        result_path = 'result_'+args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args,
                                 self.model,dsets,
                                 self.down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path)
        ap = dsets.dec_evaluation(result_path)
        return ap
