"""
Some methods are dervied and borrowed from
https://github.com/yuangh-x/2022-NIPS-Tenrec/blob/master/trainer.py
"""

import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm


def trainer(epoch, model, dataloader, optimizer, writer, args):
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    model.train()
    running_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    for data in tqdm(dataloader, total=len(dataloader), desc=f'Epoch: {epoch}, Training Batch', dynamic_ncols=True):
        optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, labels = data
        logits = model(seqs) # B x T x V
        if 'cold' in args.task_name or ('life_long' in args.task_name and args.task != 0):
            logits = logits.mean(1)
            labels = labels.view(-1)
        else:
            logits = logits.view(-1, logits.size(-1)) # (B*T) x V
            labels = labels.view(-1)  # B*T

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print(
        "Training CE Loss: {:.5f}, gpu {:d} memory".format(
            running_loss / len(dataloader),
            torch.cuda.max_memory_allocated(device=None),
        )
    )
    return optimizer


def paired_trainer(epoch, b_model, p_model, dataloader, b_optimizer, p_optimizer, writer, args):
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    b_model.train()
    p_model.train()
    running_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    for data in tqdm(dataloader, total=len(dataloader), desc=f'Epoch: {epoch}, Training Batch', dynamic_ncols=True):
        b_optimizer.zero_grad()
        p_optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, labels = data
        policy_action = p_model(seqs)
        logits = b_model(seqs, policy_action)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = loss_fn(logits, labels)
        loss.backward()
        p_optimizer.step()
        b_optimizer.step()
        running_loss += loss.detach().cpu().item()
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print(
        "Training CE Loss: {:.5f}, gpu {:d} memory".format(
            running_loss / len(dataloader),
            torch.cuda.max_memory_allocated(device=None),
        )
    )
    return b_optimizer, p_optimizer


def validator(epoch, model, dataloader, writer, args, test=False):
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    model.eval()
    avg_metrics = {}
    i = 0
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader), desc=f'Epoch: {epoch}, Validation Batch', dynamic_ncols=True):
            data = [x.to(args.device) for x in data]
            seqs, labels = data
            if test:
                scores = model.predict(seqs)
            else:
                scores = model(seqs)
            scores = scores.mean(1)
            metrics = recall_ndcg_at_k(scores, labels, args.metric_ks, args)
            i += 1
            for key, value in metrics.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value
                else:
                    avg_metrics[key] += value
    for key, value in avg_metrics.items():
        avg_metrics[key] = value / i
    print(avg_metrics)
    for k in sorted(args.metric_ks, reverse=True):
        writer.add_scalar('Val/NDCG@{}'.format(k), avg_metrics['NDCG@%d' % k], epoch)
    return avg_metrics


def paired_validator(epoch, b_model, p_model, dataloader, writer, args, test=False):
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    p_model.eval()
    b_model.eval()
    avg_metrics = {}
    i = 0
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader), desc=f'Epoch: {epoch}, Validation Batch', dynamic_ncols=True):
            data = [x.to(args.device) for x in data]
            seqs, labels = data
            policy_action = p_model(seqs)
            if test:
                scores = b_model.predict(seqs, policy_action)
            else:
                scores = b_model(seqs, policy_action)
            scores = scores[:, -1, :]  # B x V
            metrics = recall_ndcg_at_k(scores, labels, args.metric_ks, args)
            i += 1
            for key, value in metrics.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value
                else:
                    avg_metrics[key] += value
        for key, value in avg_metrics.items():
            avg_metrics[key] = value / i
        print(avg_metrics)
        for k in sorted(args.metric_ks, reverse=True):
            writer.add_scalar('Val/NDCG@{}'.format(k), avg_metrics['NDCG@%d' % k], epoch)
        return avg_metrics


def train_val_schedular(epochs, model, train_loader, val_loader, writer, args):
    if args.is_pretrain == 0:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.is_parallel:
        model = torch.nn.parallel.DistributedDataParallel(
            model,  find_unused_parameters=True,
            device_ids=[args.local_rank], output_device=args.local_rank
        )

    best_metric = all_time = val_all_time = 0

    for epoch in range(epochs):
        since = time.time()
        optimizer = trainer(epoch, model, train_loader, optimizer, writer, args)
        tmp = time.time() - since
        print("One Epoch Time: ", tmp)
        all_time += tmp
        val_since = time.time()
        metrics = validator(epoch, model, val_loader, writer, args)
        val_tmp = time.time() - val_since
        print('One epoch val: ', val_tmp)

        val_all_time += val_tmp
        if args.is_pretrain == 0 and 'acc' in args.task_name:
            if metrics['NDCG@20'] >= 0.0193:
                break
        i = 1
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            best_model = deepcopy(model)
            state_dict = model.state_dict()
            torch.save(
                state_dict, 
                os.path.join(
                    args.save_path, 
                    '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(
                        args.task_name, args.model_name, args.seed, args.is_pretrain, args.lr, args.weight_decay, 
                        args.block_num, args.hidden_size, args.embedding_size
                    )
                )
            )

        else:
            i += 1
            if i == 10:
                print('early stop!')
                break
    print('train_time:', all_time)
    print('val_time:', val_all_time)
    return best_model


def inference_acc_schedular(epochs, backbonenet, policynet, train_loader, val_loader, writer, args):
    b_optimizer = torch.optim.Adam(backbonenet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    p_optimizer = torch.optim.Adam(policynet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metric = all_time = val_all_time = 0

    for epoch in range(epochs):
        since = time.time()
        paired_trainer(epoch, backbonenet, policynet, train_loader, b_optimizer, p_optimizer, writer, args)
        tmp = time.time() - since
        print("One Epoch Time: ", tmp)
        all_time += tmp
        val_since = time.time()

        metrics = paired_validator(epoch, backbonenet, policynet, val_loader, writer, args)
        val_tmp = time.time() - val_since
        print('One epoch val: ', val_tmp)
        val_all_time += val_tmp

        i = 1
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            b_state_dict = backbonenet.state_dict()
            p_state_dict = policynet.state_dict()
            best_backbonenet = deepcopy(backbonenet)
            best_policynet = deepcopy(policynet)
            torch.save(
                b_state_dict,
                os.path.join(
                    args.save_path,
                    '{}_{}_seed{}_lr{}_block{}_best_backbone.pth'.format(
                        args.task_name, args.model_name, args.seed, args.lr, args.block_num
                    )
                )
            )
            torch.save(
                p_state_dict,
                os.path.join(
                    args.save_path,
                    '{}_{}_seed{}_lr{}_block{}_best_policynet.pth'.format(
                        args.task_name, args.model_name, args.seed, args.lr, args.block_num
                    )
                )
            )
        else:
            i += 1
            if i == 10:
                print('early stop!')
                break
    print('train_time:', all_time)
    print('val_time:', val_all_time)
    return best_backbonenet, best_policynet


def recall_ndcg_at_k(scores, labels, k_list, args):
    metrics = {}

    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(k_list, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float()).to(args.device)
        dcg = (hits * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(args.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg

    return metrics


def ctr_schedular(model, train_model_input, train_model_output, test_model_input, test_model_output, args):
    """
    Training, validation and test loop for CTR task.
    Derived from https://github.com/yuangh-x/2022-NIPS-Tenrec/blob/master/main.py
    """
    model.compile(args, "adam", "binary_crossentropy", metrics=["auc", "acc"])
    start_time = time.time()
    history, best_model = model.fit(
        train_model_input,
        train_model_output,
        batch_size=args.train_batch_size,
        epochs=args.epochs,
        verbose=2,
        validation_split=0.1111
    )
    gpu_vram = torch.cuda.max_memory_allocated(device=None)
    end_time = time.time()
    trainval_minutes = (end_time - start_time) / 60.0
    pred_ans = best_model.predict(test_model_input, args.test_batch_size)
    print("test LogLoss", round(log_loss(test_model_output, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test_model_output, pred_ans), 4))
    print(f"GPU VRAM: {gpu_vram}")
    print(f"Total training + validation time, excluding test time: {trainval_minutes} minutes.")


def mtl_schedular(model, train_dataloader, val_dataloader, test_dataloader, args):
    """
    Training, validation and test loop for MTL task.
    Derived from https://github.com/yuangh-x/2022-NIPS-Tenrec/blob/43893d187e14c0b84e0f4d889477999ee831a3c9/trainer.py#L11-L186
    """
    device = args.device
    epoch = args.epochs
    early_stop = 5
    path = os.path.join(args.save_path, '{}_{}_seed{}_best_model_{}.pth'.format(args.task_name, args.model_name, args.seed, args.mtl_task_num))
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)
    # Early stop conditions
    patience, eval_loss = 0, 0
    # Training loop
    start_time = time.time()
    if args.mtl_task_num == 2:
        print(f"[MTL] Multi-Task training started.")
        model.train()
        for i in range(epoch):
            y_train_click_true = []
            y_train_click_predict = []
            y_train_like_true = []
            y_train_like_predict = []
            total_loss, count = 0, 0
            for (x, y1, y2) in tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch: {i}, Training Batch'):
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                predict = model(x)
                y_train_click_true += list(y1.squeeze().cpu().numpy())
                y_train_like_true += list(y2.squeeze().cpu().numpy())
                y_train_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
                y_train_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
                loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
                loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
                loss = loss_1 + loss_2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
                count += 1
            click_auc = roc_auc_score(y_train_click_true, y_train_click_predict)
            like_auc = roc_auc_score(y_train_like_true, y_train_like_predict)
            print("Epoch %d train loss is %.3f, click auc is %.3f and like auc is %.3f, gpu %d memory" % (
                i + 1, total_loss / count, click_auc, like_auc, torch.cuda.max_memory_allocated(device=None)
            ))
            # Validation step
            total_eval_loss = 0
            model.eval()
            count_eval = 0
            y_val_click_true = []
            y_val_like_true = []
            y_val_click_predict = []
            y_val_like_predict = []
            for (x, y1, y2) in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Epoch: {i}, Validation Batch'):
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                predict = model(x)
                y_val_click_true += list(y1.squeeze().cpu().numpy())
                y_val_like_true += list(y2.squeeze().cpu().numpy())
                y_val_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
                y_val_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
                loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
                loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
                loss = loss_1 + loss_2
                total_eval_loss += float(loss)
                count_eval += 1
            click_auc = roc_auc_score(y_val_click_true, y_val_click_predict)
            like_auc = roc_auc_score(y_val_like_true, y_val_like_predict)
            print("Epoch %d val loss is %.3f, click auc is %.3f and like auc is %.3f, gpu %d memory" % (
                i + 1, total_eval_loss / count_eval, click_auc, like_auc, torch.cuda.max_memory_allocated(device=None)
            ))
            # Early stopping
            if i == 0:
                eval_loss = total_eval_loss / count_eval
            else:
                if total_eval_loss / count_eval < eval_loss:
                    eval_loss = total_eval_loss / count_eval
                    state = model.state_dict()
                    torch.save(state, path)
                else:
                    if patience < early_stop:
                        patience += 1
                    else:
                        print("val loss is not decrease in %d epoch and break training" % patience)
                        break
        
        end_time = time.time()
        trainval_minutes = (end_time - start_time) / 60.0
        # Test step
        print("Test loop begins.")
        state = torch.load(path)
        model.load_state_dict(state)
        total_test_loss = 0
        model.eval()
        count_eval = 0
        y_test_click_true = []
        y_test_like_true = []
        y_test_click_predict = []
        y_test_like_predict = []
        for (x, y1, y2) in tqdm(test_dataloader, total=len(test_dataloader), desc=f'Test Batch'):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_test_click_true += list(y1.squeeze().cpu().numpy())
            y_test_like_true += list(y2.squeeze().cpu().numpy())
            y_test_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_test_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            total_test_loss += float(loss)
            count_eval += 1
        click_auc = roc_auc_score(y_test_click_true, y_test_click_predict)
        like_auc = roc_auc_score(y_test_like_true, y_test_like_predict)
        print("Epoch %d test loss is %.3f, click auc is %.3f and like auc is %.3f, gpu %d memory" % (
            i + 1, total_test_loss / count_eval, click_auc, like_auc, torch.cuda.max_memory_allocated(device=None)
        ))
        print(f"Total training + validation time, excluding test time: {trainval_minutes} minutes.")

    else:
        print(f"[MTL] Single-Task training started.")
        model.train()
        for i in range(epoch):
            y_train_label_true = []
            y_train_label_predict = []
            total_loss, count = 0, 0
            for (x, y) in tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch: {i}, Training Batch'):
                x, y = x.to(device), y.to(device)
                predict = model(x)
                y_train_label_true += list(y.squeeze().cpu().numpy())
                y_train_label_predict += list(predict[0].squeeze().cpu().detach().numpy())
                loss_1 = loss_function(predict[0], y.unsqueeze(1).float())
                loss = loss_1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
                count += 1
            auc = roc_auc_score(y_train_label_true, y_train_label_predict)
            print("Epoch %d train loss is %.3f, auc is %.3f, gpu %d memory" % (
                i + 1, total_loss / count, auc, torch.cuda.max_memory_allocated(device=None)
            ))
            # Validation step
            total_eval_loss = 0
            model.eval()
            count_eval = 0
            y_val_label_true = []
            y_val_label_predict = []
            for (x, y) in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Epoch: {i}, Validation Batch'):
                x, y = x.to(device), y.to(device)
                predict = model(x)
                y_val_label_true += list(y.squeeze().cpu().numpy())
                y_val_label_predict += list(predict[0].squeeze().cpu().detach().numpy())
                loss_1 = loss_function(predict[0], y.unsqueeze(1).float())
                loss = loss_1
                total_eval_loss += float(loss)
                count_eval += 1
            auc = roc_auc_score(y_val_label_true, y_val_label_predict)
            print("Epoch %d val loss is %.3f, auc is %.3f, gpu %d memory" % (
                i + 1, total_eval_loss / count_eval, auc, torch.cuda.max_memory_allocated(device=None)
            ))
            # Early stopping
            if i == 0:
                eval_loss = total_eval_loss / count_eval
            else:
                if total_eval_loss / count_eval < eval_loss:
                    eval_loss = total_eval_loss / count_eval
                    state = model.state_dict()
                    torch.save(state, path)
                else:
                    if patience < early_stop:
                        patience += 1
                    else:
                        print("val loss is not decrease in %d epoch and break training" % patience)
                        break

        end_time = time.time()
        trainval_minutes = (end_time - start_time) / 60.0
        # Test step
        total_test_loss = 0
        model.eval()
        count_eval = 0
        y_test_label_true = []
        y_test_label_predict = []
        for (x, y) in tqdm(test_dataloader, total=len(test_dataloader), desc=f'Test Batch'):
            x, y = x.to(device), y.to(device)
            predict = model(x)
            y_test_label_true += list(y.squeeze().cpu().numpy())
            y_test_label_predict += list(predict[0].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y.unsqueeze(1).float())
            loss = loss_1
            total_test_loss += float(loss)
            count_eval += 1
        auc = roc_auc_score(y_test_label_true, y_test_label_predict)
        print("Test loss is %.3f, auc is %.3f, gpu %d memory" % (
            total_test_loss / count_eval, auc, torch.cuda.max_memory_allocated(device=None)
        ))
        print(f"Total training + validation time, excluding test time: {trainval_minutes} minutes.")
