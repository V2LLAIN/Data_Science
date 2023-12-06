import tqdm, wandb, gc
import math, time, torch
import torch.nn as nn
import numpy as np
import pandas as pd
from config import args
from util import LOGGER, get_score, sigmoid, get_optimizer_params
from dataset import collate, CustomDataset, device, train_df
from model import CustomModel
from tqdm.auto import tqdm
from tokenizer import tokenizer
from torch.utils.data import DataLoader

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """One epoch training pass."""
    model.train()  # set model in train mode
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.APEX)  # Automatic Mixed Precision tries to match each op to its appropriate datatype.
    losses = AverageMeter()  # initiate AverageMeter to track the loss.
    start = end = time.time()  # track the execution time.
    global_step = 0

    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, batch in enumerate(tqdm_train_loader):
            inputs = batch.pop("inputs")
            labels = batch.pop("labels")
            inputs = collate(inputs)  # collate inputs
            for k, v in inputs.items():  # send each tensor value to `device`
                inputs[k] = v.to(device)
            labels = labels.to(device)  # send labels to `device`
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=args.APEX):
                y_preds = model(inputs)  # forward propagation pass
                loss = criterion(y_preds, labels.unsqueeze(1))  # get loss
            if args.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / args.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)  # update loss function tracking
            scaler.scale(loss).backward()  # backward propagation pass
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.MAX_GRAD_NORM)

            if (step + 1) % args.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)  # update optimizer parameters
                scaler.update()
                optimizer.zero_grad()  # zero out the gradients
                global_step += 1
                if args.BATCH_SCHEDULER:
                    scheduler.step()  # update learning rate
            end = time.time()  # get finish time

            # ========== LOG INFO ==========
            if step % args.PRINT_FREQ == 0 or step == (len(train_loader) - 1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch + 1, step, len(train_loader),
                              remain=timeSince(start, float(step + 1) / len(train_loader)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_lr()[0]))
            if args.WANDB:
                wandb.log({f"[fold_{fold}] train loss": losses.val,
                           f"[fold_{fold}] lr": scheduler.get_lr()[0]})

    return losses.avg


def valid_epoch(valid_loader, model, criterion, device):
    model.eval()  # set model in evaluation mode
    losses = AverageMeter()  # initiate AverageMeter for tracking the loss.
    prediction_dict = {}
    preds = []
    start = end = time.time()  # track the execution time.
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, batch in enumerate(tqdm_valid_loader):
            inputs = batch.pop("inputs")
            labels = batch.pop("labels")
            ids = batch.pop("ids")
            inputs = collate(inputs)  # collate inputs
            for k, v in inputs.items():
                inputs[k] = v.to(device)  # send inputs to device
            labels = labels.to(device)
            batch_size = labels.size(0)
            with torch.no_grad():
                y_preds = model(inputs)  # forward propagation pass
                loss = criterion(y_preds, labels.unsqueeze(1))  # get loss
            if args.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / args.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)  # update loss function tracking
            preds.append(y_preds.to('cpu').numpy())  # save predictions
            end = time.time()  # get finish time

            # ========== LOG INFO ==========
            if step % args.PRINT_FREQ == 0 or step == (len(valid_loader) - 1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(valid_loader),
                              loss=losses,
                              remain=timeSince(start, float(step + 1) / len(valid_loader))))
            if args.WANDB:
                wandb.log({f"[fold_{fold}] val loss": losses.val})

    prediction_dict["predictions"] = np.concatenate(preds)  # np.array() of shape (fold_size, target_cols)
    prediction_dict["ids"] = ids
    return losses.avg, prediction_dict


def train_loop(folds, fold):
    LOGGER.info(f"========== Fold: {fold} training ==========")

    # ======== SPLIT ==========
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    valid_labels = valid_folds['generated'].values

    # ======== DATASETS ==========
    train_dataset = CustomDataset(args, train_folds, tokenizer)
    valid_dataset = CustomDataset(args, valid_folds, tokenizer)

    # ======== DATALOADERS ==========
    train_loader = DataLoader(train_dataset,
                              batch_size=args.BATCH_SIZE_TRAIN,  # TODO: split into train and valid
                              shuffle=True,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.BATCH_SIZE_VALID,
                              shuffle=False,
                              pin_memory=True, drop_last=False)

    # ======== MODEL ==========
    model = CustomModel(args, config_path=None, pretrained=True)
    torch.save(model.config, args.OUTPUT_DIR + '/config.pth')
    model.to(device)

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=args.ENCODER_LR,
                                                decoder_lr=args.DECODER_LR,
                                                weight_decay=args.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(optimizer_parameters,
                      lr=args.ENCODER_LR,
                      eps=args.EPS,
                      betas=args.BETAS)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-5,
        epochs=args.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )

    # ======= LOSS ==========
    criterion = nn.BCEWithLogitsLoss()

    best_score = -np.inf
    # ====== ITERATE EPOCHS ========
    for epoch in range(args.EPOCHS):

        start_time = time.time()

        # ======= TRAIN ==========
        avg_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # ======= EVALUATION ==========
        avg_val_loss, prediction_dict = valid_epoch(valid_loader, model, criterion, device)
        predictions = prediction_dict["predictions"]
        # ======= SCORING ==========
        score = get_score(valid_labels, sigmoid(predictions))

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')

        if args.WANDB:
            wandb.log({f"[fold_{fold}] epoch": epoch + 1,
                       f"[fold_{fold}] avg_train_loss": avg_loss,
                       f"[fold_{fold}] avg_val_loss": avg_val_loss,
                       f"[fold_{fold}] score": score})

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(),
                       args.OUTPUT_DIR + f"/{args.MODEL.replace('/', '_')}_fold_{fold}_best.pth")
            best_model_predictions = predictions

    valid_folds["preds"] = best_model_predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


if __name__ == '__main__':
    def get_result(oof_df):
        labels = oof_df["generated"].values
        preds = oof_df["preds"].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')


    if args.TRAIN:
        oof_df = pd.DataFrame()
        for fold in range(args.FOLDS):
            if fold == 0:
                _oof_df = train_loop(train_df, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== Fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_csv(args.OUTPUT_DIR + '/oof_df.csv', index=False)
    if args.WANDB:
        wandb.finish()