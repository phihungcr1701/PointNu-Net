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

# Resume training arguments (for multi-session training on Kaggle)
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Full path to model checkpoint (model_best.pt or model_last.pt)')
parser.add_argument('--optimizer_path', type=str, default=None, help='Full path to optimizer state file')
parser.add_argument('--scheduler_path', type=str, default=None, help='Full path to scheduler state file')
parser.add_argument('--training_log_path', type=str, default=None, help='Full path to training.log file')
parser.add_argument('--start_epoch', type=int, default=None, help='Epoch to resume from (if not provided, will be read from log file)')
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
    val_interval = opts.val_interval if opts.val_interval is not None else config['train'].get('val_interval', 1)
    print(f"\u2713 Validation interval: every {val_interval} epoch(s)")
    
    # Initialize training state
    start_epoch = 0
    best_mPQ = -1
    best_epoch = -1
    logger = None
    
    # ===== RESUME LOGIC =====
    if opts.resume:
        print("\n" + "="*70)
        print("RESUME MODE: Multi-Session Training")
        print("="*70)
        
        # Validate all required paths are provided
        required_args = {
            'checkpoint_path': opts.checkpoint_path,
            'optimizer_path': opts.optimizer_path,
            'scheduler_path': opts.scheduler_path,
            'training_log_path': opts.training_log_path
        }
        
        for arg_name, arg_value in required_args.items():
            if arg_value is None:
                raise ValueError(f"--{arg_name} is required when using --resume")
            if not os.path.exists(arg_value):
                raise FileNotFoundError(f"{arg_name} not found: {arg_value}")
            print(f"\u2713 {arg_name}: {arg_value}")
        
        # Load model, optimizer, scheduler
        print("\n\u2192 Loading checkpoint...")
        trainer.load_checkpoint(opts.checkpoint_path)
        trainer.load_optimizer(opts.optimizer_path)
        trainer.load_scheduler(opts.scheduler_path)
        
        # Parse training log to get start_epoch and best metrics
        print("\n\u2192 Parsing training log...")
        try:
            with open(opts.training_log_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 1:  # Has header + data
                # Get start_epoch from explicit argument or from log
                if opts.start_epoch is not None:
                    start_epoch = opts.start_epoch
                    print(f"\u2713 Start epoch from argument: {start_epoch}")
                else:
                    # Parse last line to get last completed epoch
                    last_line = lines[-1].strip()
                    parts = last_line.split(',')
                    if len(parts) >= 1:
                        last_completed_epoch = int(parts[0])
                        start_epoch = last_completed_epoch + 1
                        print(f"\u2713 Last completed epoch: {last_completed_epoch}")
                        print(f"\u2713 Resuming from epoch: {start_epoch}")
                
                # Scan log to find best mPQ (column 4 in CSV)
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split(',')
                    if len(parts) >= 5 and parts[4]:  # Has val_mPQ
                        try:
                            val_mPQ = float(parts[4])
                            epoch_num = int(parts[0])
                            if val_mPQ > best_mPQ:
                                best_mPQ = val_mPQ
                                best_epoch = epoch_num
                        except:
                            pass
                
                if best_mPQ > -1:
                    print(f"\u2713 Best metrics restored: epoch={best_epoch}, mPQ={best_mPQ:.6f}")
                else:
                    print(f"\u26a0 WARNING: No validation metrics found in log")
            else:
                print(f"\u26a0 WARNING: Log file is empty")
                start_epoch = 0
                
        except Exception as e:
            print(f"\u26a0 ERROR parsing log: {e}")
            start_epoch = 0
            best_mPQ = -1
        
        print("="*70 + "\n")
    
    # Initialize logger (append mode if resume, else create new)
    logger = TrainingLogger(output_directory, resume=opts.resume, log_file_path=opts.training_log_path if opts.resume else None)
    if opts.resume:
        print(f"\u2713 Logger initialized in APPEND mode\n")
    
    iter_per_epoch = len(train_loader)
    max_epoch = config['train']['max_epoch']
    
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Epochs: {start_epoch} → {max_epoch-1} ({max_epoch - start_epoch} total)")
    print(f"Best mPQ: {best_mPQ if best_mPQ > -1 else 'N/A'} @ epoch {best_epoch if best_epoch > -1 else 'N/A'}")
    print(f"{'='*70}\n")
    
    # ===== TRAINING LOOP =====
    for epoch in range(start_epoch, max_epoch):
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
                    trainer.save_checkpoint(checkpoint_directory, name='best')
                    print(f"\n✓ NEW BEST MODEL at epoch {epoch}: mPQ={val_mPQ:.6f}")
                    if opts.resume:
                        print(f"  → Checkpoint files ready for next session: model_best.pt, optimizer.pt, scheduler.pt")
        
        # --- Logging ---
        logger.log(epoch, iter_per_epoch, train_loss_ins_avg, train_loss_cate_avg,
                   val_mPQ, val_bPQ, is_best)
        
        # --- Save last model (always) ---
        trainer.save_checkpoint(checkpoint_directory, name='last')
        
        print(f"\n[Epoch {epoch}] train_loss_ins: {train_loss_ins_avg:.6f}, train_loss_cate: {train_loss_cate_avg:.6f}")
    
    # ===== TRAINING COMPLETE =====
    logger.close()
    print(f"\n{'='*70}")
    print(f"✓ TRAINING SESSION COMPLETE")
    print(f"{'='*70}")
    
    if opts.resume:
        print(f"\n[MULTI-SESSION INFO]")
        print(f"  Session type: RESUME")
        print(f"  Epochs completed: {start_epoch} → {max_epoch-1}")
        print(f"  Total epochs so far: {max_epoch}")
        print(f"  Next session: Resume from epoch {max_epoch}")
    else:
        print(f"\n[NEW TRAINING SESSION]")
        print(f"  Total epochs: {max_epoch}")
    
    print(f"\n[BEST MODEL]")
    if best_epoch >= 0:
        print(f"  Epoch: {best_epoch}")
        print(f"  mPQ: {best_mPQ:.6f}")
    else:
        print(f"  No validation performed")
    
    print(f"\n[CHECKPOINT LOCATIONS]")
    print(f"  model_best.pt: {os.path.join(checkpoint_directory, 'model_best.pt')}")
    print(f"  model_last.pt: {os.path.join(checkpoint_directory, 'model_last.pt')}")
    print(f"  optimizer.pt: {os.path.join(checkpoint_directory, 'optimizer.pt')}")
    print(f"  scheduler.pt: {os.path.join(checkpoint_directory, 'scheduler.pt')}")
    print(f"  training.log: {os.path.join(output_directory, 'training.log')}")
    
    if opts.resume or max_epoch > 30:
        print(f"\n[NEXT STEP: CONTINUE TRAINING]")
        print(f"  To resume in next session, use:")
        print(f"  python train_pannuke.py \\")
        print(f"    --name {opts.name} \\")
        print(f"    --train_fold {opts.train_fold} \\")
        print(f"    --val_fold {opts.val_fold} \\")
        print(f"    --test_fold {opts.test_fold} \\")
        print(f"    --resume \\")
        print(f"    --checkpoint_path {os.path.join(checkpoint_directory, 'model_best.pt')} \\")
        print(f"    --optimizer_path {os.path.join(checkpoint_directory, 'optimizer.pt')} \\")
        print(f"    --scheduler_path {os.path.join(checkpoint_directory, 'scheduler.pt')} \\")
        print(f"    --training_log_path {os.path.join(output_directory, 'training.log')}")
    
    print(f"\n{'='*70}\n")





