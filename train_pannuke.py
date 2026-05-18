from utils.dataloader import NucleiDataset,PannukeDataset
from trainer import Trainer
from torch.utils.data import DataLoader
import sys
from utils import prepare_sub_folder,get_config,collate_func,TrainingLogger,run_validation
import torch
import numpy as np
import os,shutil
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/pannuke.yaml')
parser.add_argument('--name', type=str, default='pannuke_experiment')
parser.add_argument('--train_fold', type=int, default=2)
parser.add_argument('--val_fold', type=int, default=1)
parser.add_argument('--test_fold', type=int, default=3)
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--val_interval', type=int, default=None)  # Override config if provided
opts = parser.parse_args()

def check_manual_seed(seed):
    """ If manual seed is not specified, choose a
    random one and communicate it to the user.
    Args:
        seed: seed to check
    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ia.random.seed(seed)

    print("Using manual seed: {seed}".format(seed=seed))
    return

if __name__ == '__main__':
    config=get_config(opts.config)
    check_manual_seed(opts.seed)
    
    # ===== LOAD DATASETS =====
    train_dataset=PannukeDataset(data_root=config['dataroot'], seed=opts.seed, is_train=True, fold=opts.train_fold,output_stride=config['model']['output_stride'])
    val_dataset=PannukeDataset(data_root=config['dataroot'], seed=opts.seed, is_train=False, fold=opts.val_fold,output_stride=config['model']['output_stride'])
    
    train_loader=DataLoader(dataset=train_dataset, batch_size=config['train']['batch_size'], shuffle=True, drop_last=True, num_workers=config['train']['num_workers'],persistent_workers=True,collate_fn=collate_func,pin_memory=True)

    output_directory = os.path.join(opts.output_dir, opts.name, 'train_{}_to_test_{}'.format( opts.train_fold,opts.test_fold))
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config,os.path.join(output_directory,'config.yaml'))

    # ===== SETUP TRAINING =====
    trainer = Trainer(config)
    trainer.cuda()
    
    # Get validation interval from config or argument
    val_interval = opts.val_interval if opts.val_interval is not None else config['train'].get('val_interval', 5)
    print(f"Validation interval: every {val_interval} epochs")
    
    # Initialize logger
    logger = TrainingLogger(output_directory)
    
    best_mPQ = -1
    best_epoch = -1
    iter_per_epoch = len(train_loader)
    max_epoch = config['train']['max_epoch']
    
    # ===== TRAINING LOOP =====
    for epoch in range(max_epoch):
        # --- Training phase ---
        train_loss_ins_list = []
        train_loss_cate_list = []
        
        for i, train_data in enumerate(train_loader, start=1):
            for k in train_data.keys():
                if not isinstance(train_data[k],list):
                    train_data[k]=train_data[k].cuda().detach()
                else:
                    train_data[k] = [s.cuda().detach() if s is not None else s for s in train_data[k]]
            
            ins_loss, cate_loss, maskiou_loss = trainer.seg_updata(train_data)
            train_loss_ins_list.append(ins_loss)
            train_loss_cate_list.append(cate_loss)
            
            sys.stdout.write(f'\r epoch:{epoch}/{max_epoch-1} step:{i}/{iter_per_epoch} ins_loss: {ins_loss:.4f} cate_loss: {cate_loss:.4f}')
            sys.stdout.flush()
        
        # Compute average losses for this epoch
        train_loss_ins_avg = np.mean(train_loss_ins_list)
        train_loss_cate_avg = np.mean(train_loss_cate_list)
        
        trainer.scheduler.step()
        
        # --- Validation phase (every val_interval epochs) ---
        val_mPQ = None
        val_bPQ = None
        is_best = 0
        
        if (epoch + 1) % val_interval == 0:
            val_mPQ, val_bPQ = run_validation(
                trainer, val_dataset, opts, checkpoint_directory, epoch, output_directory, config
            )
            
            if val_mPQ is not None:
                is_best = 1 if val_mPQ > best_mPQ else 0
                
                if is_best:
                    best_mPQ = val_mPQ
                    best_epoch = epoch
                    trainer.save(checkpoint_directory, 'best')
                    print(f"\n✓ NEW BEST MODEL at epoch {epoch}: mPQ={val_mPQ:.4f}")
        
        # --- Logging ---
        logger.log(epoch, iter_per_epoch, train_loss_ins_avg, train_loss_cate_avg,
                   val_mPQ, val_bPQ, is_best)
        
        # --- Save last model ---
        trainer.save(checkpoint_directory, 'last')
        
        print(f"\n[Epoch {epoch}] train_loss_ins: {train_loss_ins_avg:.6f}, train_loss_cate: {train_loss_cate_avg:.6f}")
    
    # ===== TRAINING COMPLETE =====
    logger.close()
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best epoch: {best_epoch}, Best mPQ: {best_mPQ:.4f}" if best_epoch >= 0 else "No validation performed")
    print(f"Models saved in: {checkpoint_directory}")
    print(f"Logs saved in: {os.path.join(output_directory, 'training.log')}")
    print(f"{'='*60}")





