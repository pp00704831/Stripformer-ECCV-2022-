import logging
from functools import partial
import os
import cv2
import torch
import torch.optim as optim
import tqdm
import yaml
from joblib import cpu_count
from torch.utils.data import DataLoader
import random
from dataset import PairedDataset
from metric_counter import MetricCounter
from models.losses import get_loss
from models.models import get_model
from models.networks import get_nets
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
cv2.setNumThreads(0)


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.metric_counter = MetricCounter(config['experiment_desc'])

    def train(self):
        self._init_params()
        start_epoch = 0
        if os.path.exists('last_Stripformer_pretrained.pth'):
            print('load_pretrained')
            training_state = (torch.load('last_Stripformer_pretrained.pth'))
            start_epoch = training_state['epoch']
            new_weight = self.netG.state_dict()
            new_weight.update(training_state['model_state'])
            self.netG.load_state_dict(new_weight)
            new_optimizer = self.optimizer_G.state_dict()
            new_optimizer.update(training_state['optimizer_state'])
            self.optimizer_G.load_state_dict(new_optimizer)
            new_scheduler = self.scheduler_G.state_dict()
            new_scheduler.update(training_state['scheduler_state'])
            self.scheduler_G.load_state_dict(new_scheduler)


        for epoch in range(start_epoch, config['num_epochs']):
            self._run_epoch(epoch)
            if epoch % 30 == 0 or epoch == (config['num_epochs']-1):
                self._validate(epoch)
            self.scheduler_G.step()

            scheduler_state = self.scheduler_G.state_dict()
            training_state = {'epoch': epoch,  'model_state': self.netG.state_dict(),
                              'scheduler_state': scheduler_state, 'optimizer_state': self.optimizer_G.state_dict()}
            if self.metric_counter.update_best_model():
                torch.save(training_state['model_state'], 'best_{}.pth'.format(self.config['experiment_desc']))

            if epoch % 300 == 0:
                torch.save(training_state, 'last_{}_{}.pth'.format(self.config['experiment_desc'], epoch))

            if epoch == (config['num_epochs']-1):
                torch.save(training_state['model_state'], 'final_{}.pth'.format(self.config['experiment_desc']))

            torch.save(training_state, 'last_{}.pth'.format(self.config['experiment_desc']))
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            outputs = self.netG(inputs)
            self.optimizer_G.zero_grad()
            loss_G = self.criterionG(outputs, targets, inputs)
            loss_G.backward()
            self.optimizer_G.step()
            self.metric_counter.add_losses(loss_G.item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            with torch.no_grad():
                inputs, targets = self.model.get_input(data)
                outputs = self.netG(inputs)
                loss_G = self.criterionG(outputs, targets, inputs)
                self.metric_counter.add_losses(loss_G.item())
                curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
                self.metric_counter.add_metrics(curr_psnr, curr_ssim)
                if not i:
                    self.metric_counter.add_image(img_for_vis, tag='val')
                i += 1
                if i > epoch_size:
                    break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)


    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'], eta_min=self.config['scheduler']['min_lr'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    def _init_params(self):
        self.criterionG = get_loss(self.config['model'])
        self.netG = get_nets(self.config['model'])
        self.netG.cuda()
        self.model = get_model(self.config['model'])
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.scheduler_G = self._get_scheduler(self.optimizer_G)


if __name__ == '__main__':
    with open('config/config_Stripformer_pretrained.yaml', 'r') as f:
        config = yaml.load(f)
    # setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # set random seed
    seed = 666
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(), shuffle=True, drop_last=False)

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(PairedDataset.from_config, datasets)
    train, val = map(get_dataloader, datasets)
    trainer = Trainer(config, train=train, val=val)
    trainer.train()
