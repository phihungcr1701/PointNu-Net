import os
import yaml
import csv
from datetime import datetime
import subprocess
import shutil
import sys
import numpy as np
import torch
import re

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF','npy','mat'
]

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory

def get_config(config):
    with open(config,'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class TrainingLogger:
    """Logger untuk training statistics (loss + validation metrics)"""

    def __init__(self, output_dir, resume=False, log_file_path=None):
        """
        Initialize logger.

        Args:
            output_dir: directory to save training.log
            resume: if True, append to existing log
            log_file_path: external path to old log file
        """

        os.makedirs(output_dir, exist_ok=True)

        # ALWAYS use writable output directory
        self.log_file = os.path.join(output_dir, 'training.log')

        self.file_handle = None
        self.csv_writer = None
        self.resume = resume

        # ===== RESUME MODE =====
        if resume:

            # Copy old log to writable location
            if log_file_path is not None:

                if not os.path.exists(log_file_path):
                    raise FileNotFoundError(
                        f"Resume log file not found: {log_file_path}"
                    )

                # Copy only if destination doesn't exist
                if not os.path.exists(self.log_file):
                    shutil.copy(log_file_path, self.log_file)
                    print(f"✓ Copied old log:")
                    print(f"  from: {log_file_path}")
                    print(f"  to:   {self.log_file}")

            # Open copied log in append mode
            if os.path.exists(self.log_file):
                self.file_handle = open(self.log_file, 'a', newline='')
                self.csv_writer = csv.writer(self.file_handle)

                print(f"✓ Logger appending to: {self.log_file}")

            else:
                print("⚠ No existing log found, creating new log")
                self._write_header()

        # ===== NEW TRAINING =====
        else:
            self._write_header()

    def _write_header(self):
        """Write CSV header"""

        self.file_handle = open(self.log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.file_handle)

        header = [
            'epoch',
            'step',
            'train_loss_ins',
            'train_loss_cate',
            'val_mPQ',
            'val_bPQ',
            'is_best',
            'timestamp'
        ]

        self.csv_writer.writerow(header)
        self.file_handle.flush()

    def log(self, epoch, num_steps, train_loss_ins, train_loss_cate,
            val_mPQ=None, val_bPQ=None, is_best=0):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        val_mPQ_str = f"{val_mPQ:.6f}" if val_mPQ is not None else ""
        val_bPQ_str = f"{val_bPQ:.6f}" if val_bPQ is not None else ""

        row = [
            epoch,
            num_steps,
            f"{train_loss_ins:.6f}",
            f"{train_loss_cate:.6f}",
            val_mPQ_str,
            val_bPQ_str,
            is_best,
            timestamp
        ]

        self.csv_writer.writerow(row)
        self.file_handle.flush()

    def close(self):
        """Close logger"""

        if self.file_handle is not None:
            self.file_handle.close()


def run_validation(trainer, val_dataset, opts, checkpoint_dir, epoch, output_dir, config):
    """
    Run validation on validation fold using scripts (infer_pannuke.py + PanNuKe-metrics/run.py).
    
    Args:
        trainer: Trainer object
        val_dataset: validation dataset
        opts: argparse options (name, train_fold, val_fold, test_fold)
        checkpoint_dir: checkpoint directory
        epoch: current epoch
        output_dir: output directory for experiment
        config: config dict (contains dataroot, model params)
    
    Returns:
        mPQ, bPQ (floats) or (None, None) if validation fails
    """
    try:
        from torch.utils.data import DataLoader
        from utils import collate_func
        
        print(f"\n[Epoch {epoch}] Running validation inference...")
        
        # Step 1: Run inference on validation fold
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=1, 
            shuffle=False, 
            drop_last=False, 
            num_workers=0,
            collate_fn=collate_func,
            pin_memory=True
        )
        
        predictions = []
        trainer.model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                for k in val_data.keys():
                    if not isinstance(val_data[k], list):
                        val_data[k] = val_data[k].cuda().detach()
                    else:
                        val_data[k] = [s.cuda().detach() if s is not None else s for s in val_data[k]]
                
                img = val_data['image']
                output = trainer.prediction(img, score_thr=0.4, update_thr=0.2)
                
                if output is not None:
                    seg_masks, cate_labels, cate_scores = output
                    seg_masks = seg_masks.cpu().numpy()
                    cate_labels = cate_labels.cpu().numpy()
                    pred = _stack_prediction(seg_masks, cate_labels)
                else:
                    pred = np.zeros((256, 256, 6))
                
                predictions.append(pred)
        
        trainer.model.train()
        
        # Save predictions as masks_val.npy
        predictions = np.stack(predictions, 0).astype(np.int16)
        masks_val_path = os.path.join(output_dir, 'masks_val.npy')
        np.save(masks_val_path, predictions)
        print(f"[Epoch {epoch}] Validation predictions saved to {masks_val_path}")
        print(f"[Epoch {epoch}] Predictions stats: shape={predictions.shape}, dtype={predictions.dtype}, min={predictions.min()}, max={predictions.max()}, non_zero={np.count_nonzero(predictions)}")
        
        # Step 2: Run PanNuKe-metrics/run.py to compute metrics
        print(f"[Epoch {epoch}] Computing validation metrics using PanNuKe-metrics/run.py...")
        
        # Find GT masks file and types.npy
        try:
            from utils.dataloader import _resolve_pannuke_file
            gt_masks_path = _resolve_pannuke_file(config['dataroot'], opts.val_fold, 'masks.npy')
            gt_folder = os.path.dirname(gt_masks_path)
            print(f"[Epoch {epoch}] GT masks path: {gt_masks_path}")
        except FileNotFoundError as e:
            print(f"[Epoch {epoch}] Could not find GT masks: {e}")
            return None, None
        
        # Find types.npy from images folder
        types_candidates = [
            os.path.join(gt_folder, '..', '..', 'images', os.path.basename(gt_folder), 'types.npy'),
            os.path.join(gt_folder, '..', '..', '..', 'images', os.path.basename(gt_folder), 'types.npy'),
        ]
        types_path = None
        for candidate in types_candidates:
            candidate = os.path.normpath(candidate)
            if os.path.exists(candidate):
                types_path = candidate
                print(f"[Epoch {epoch}] Found types.npy: {types_path}")
                break
        
        if types_path is None:
            print(f"[Epoch {epoch}] ERROR: Could not find types.npy")
            return None, None
        
        # Check if we should save per-epoch metrics CSV
        save_csv = config.get('train', {}).get('save_val_metrics_csv', False)
        
        if save_csv:
            metrics_save_dir = os.path.join(output_dir, f'val_metrics_epoch{epoch}')
            print(f"[Epoch {epoch}] CSV flag enabled - will save metrics to: {metrics_save_dir}")
        else:
            metrics_save_dir = os.path.join(output_dir, 'val_metrics_tmp')
            print(f"[Epoch {epoch}] CSV flag disabled - using temp metrics folder (will be deleted)")
        
        os.makedirs(metrics_save_dir, exist_ok=True)
        
        # Call run.py with full file paths
        cmd = [
            'python', 'PanNuKe-metrics/run.py',
            '--true_path', gt_masks_path,         # ← Full path to GT masks.npy file
            '--pred_path', masks_val_path,        # ← Full path to predictions masks_val.npy file
            '--type_path', types_path,            # ← Full path to types.npy file
            '--save_path', metrics_save_dir
        ]
        
        env = os.environ.copy()
        # Add PanNuKe-metrics to PYTHONPATH so it can import utils.py from its folder
        env['PYTHONPATH'] = os.path.abspath('PanNuKe-metrics')
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"[Epoch {epoch}] run.py failed with return code {result.returncode}")
            print(f"[Epoch {epoch}] stderr: {result.stderr}")
            print(f"[Epoch {epoch}] stdout: {result.stdout}")
            # Clean up metrics folder if needed
            if not save_csv:
                import shutil
                shutil.rmtree(metrics_save_dir, ignore_errors=True)
            return None, None
        
        # Step 3: Parse output to extract mPQ, bPQ
        print(f"[Epoch {epoch}] run.py stdout:\n{result.stdout}")
        mPQ, bPQ = _parse_pannuke_metrics(result.stdout)
        
        # Clean up metrics temp folder if flag is disabled
        if not save_csv:
            import shutil
            shutil.rmtree(metrics_save_dir, ignore_errors=True)
            print(f"[Epoch {epoch}] Temporary metrics folder deleted (save_val_metrics_csv=false)")
        
        if mPQ is not None and bPQ is not None:
            print(f"[Epoch {epoch}] Validation: mPQ={mPQ:.4f}, bPQ={bPQ:.4f}")
            if save_csv:
                print(f"[Epoch {epoch}] Metrics CSV saved to: {metrics_save_dir}/")
                print(f"  - class_stats.csv: 5 nuclei classes")
                print(f"  - tissue_stats.csv: 19 tissues + mean row")
            return mPQ, bPQ
        else:
            print(f"[Epoch {epoch}] Failed to parse metrics from run.py output")
            print(f"[Epoch {epoch}] run.py stdout:\n{result.stdout}")
            return None, None
    
    except Exception as e:
        print(f"[Epoch {epoch}] Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _stack_prediction(seg_masks, cate_labels):
    """Convert segmentation masks to 6-channel format (PanNuke format)"""
    out_seg = np.zeros((256, 256, 6))
    idx_num = 1
    for mask, label in zip(seg_masks, cate_labels):
        assert label != 5
        out_seg[:, :, label] = np.maximum(out_seg[:, :, label], mask * idx_num)
        idx_num += 1
    out_seg[:, :, 5] = np.sum(out_seg[:, :, :5], axis=-1) == 0
    return out_seg


def _parse_pannuke_metrics(stdout_text):
    """
    Parse mPQ and bPQ from PanNuKe-metrics/run.py stdout output.
    
    Example output lines:
        Average mPQ:0.5234
        Average bPQ:0.6123
    
    Returns:
        (mPQ, bPQ) tuple or (None, None) if parsing fails
    """
    try:
        mPQ = None
        bPQ = None
        
        # Search for "Average mPQ:" line
        mpq_match = re.search(r'Average mPQ:([\d.]+)', stdout_text)
        if mpq_match:
            mPQ = float(mpq_match.group(1))
        
        # Search for "Average bPQ:" line
        bpq_match = re.search(r'Average bPQ:([\d.]+)', stdout_text)
        if bpq_match:
            bPQ = float(bpq_match.group(1))
        
        return mPQ, bPQ
    
    except Exception as e:
        print(f"Error parsing metrics: {e}")
        print(f"Stdout was: {stdout_text}")
        return None, None
