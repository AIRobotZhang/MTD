# -*- coding:utf-8 -*-
import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import pickle as pkl
from torch.nn import MSELoss

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertConfig
)

from models.modeling_span import Span_Detector
from models.modeling_type import Type_Classifier
from utils.data_utils import load_and_cache_examples, get_labels
from utils.eval import evaluate
from utils.config import config
from utils.loss_utils import share_loss

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "span": (Span_Detector, BertConfig, BertTokenizer),
    "type": (Type_Classifier, BertConfig, BertTokenizer),
}

torch.set_printoptions(profile="full")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, tokenizer, t_total, span_num_labels, type_num_labels_src, type_num_labels_tgt):
    model_class, config_class, _ = MODEL_CLASSES["span"]

    config = config_class.from_pretrained(
        args.span_model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model = model_class.from_pretrained(
        args.span_model_name_or_path,
        config=config,
        span_num_labels=span_num_labels,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    span_model.to(args.device)
    
    model_class, config_class, _ = MODEL_CLASSES["type"]

    config = config_class.from_pretrained(
        args.type_model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    type_model = model_class.from_pretrained(
        args.type_model_name_or_path,
        config=config,
        type_num_labels_src=type_num_labels_src,
        type_num_labels_tgt=type_num_labels_tgt, 
        device=args.device,
        domain=args.dataset,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    type_model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters_span = [
        {
            "params": [p for n, p in span_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in span_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_span = AdamW(optimizer_grouped_parameters_span, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_span = get_linear_schedule_with_warmup(
        optimizer_span, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    optimizer_grouped_parameters_type = [
        {
            "params": [p for n, p in type_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in type_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_type = AdamW(optimizer_grouped_parameters_type, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler_type = get_linear_schedule_with_warmup(
        optimizer_type, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        span_model = torch.nn.DataParallel(span_model)
        type_model = torch.nn.DataParallel(type_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        span_model = torch.nn.parallel.DistributedDataParallel(
            span_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        type_model = torch.nn.parallel.DistributedDataParallel(
            type_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    span_model.zero_grad()
    type_model.zero_grad()

    return span_model, type_model, optimizer_span, scheduler_span, optimizer_type, scheduler_type

def validation(args, span_model, type_model, tokenizer, id_to_label_span, pad_token_label_id, best_dev, test, best_dev_bio, test_bio,\
         global_step, t_total, epoch, devs, tests):
    best_dev, best_dev_bio, is_updated_dev = evaluate(devs, args, span_model, type_model, tokenizer, \
        id_to_label_span, pad_token_label_id, best_dev, best_dev_bio, mode="dev", logger=logger, \
        prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
    test, test_bio, _ = evaluate(tests, args, span_model, type_model, tokenizer, \
        id_to_label_span, pad_token_label_id, test, test_bio, mode="test", logger=logger, \
        prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
    
    if args.local_rank in [-1, 0] and is_updated_dev:
        path = os.path.join(args.output_dir, "checkpoint-best-span-dev")
        logger.info("Saving span model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
            span_model.module if hasattr(span_model, "module") else span_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)

        path = os.path.join(args.output_dir, "checkpoint-best-type-dev")
        logger.info("Saving type model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
            type_model.module if hasattr(type_model, "module") else type_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)

        tokenizer.save_pretrained(path)

    return best_dev, test, best_dev_bio, test_bio, is_updated_dev

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train(args, train_dataset_src, train_dataset, id_to_label_span, id_to_label_type_src, id_to_label_type_tgt, tokenizer, pad_token_label_id):
    """ Train the model """
    # num_labels = len(labels)
    span_num_labels = len(id_to_label_span)
    type_num_labels_src = len(id_to_label_type_src)-1
    type_num_labels_tgt = len(id_to_label_type_tgt)-1
    args.train_batch_size_src = args.per_gpu_train_batch_size_src * max(1, args.n_gpu)
    train_sampler_src = RandomSampler(train_dataset_src) if args.local_rank==-1 else DistributedSampler(train_dataset_src)
    train_dataloader_src = DataLoader(train_dataset_src, sampler=train_sampler_src, batch_size=args.train_batch_size_src)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(len(train_dataloader_src)//args.gradient_accumulation_steps)+1
    else:
        t_total = len(train_dataloader_src)//args.gradient_accumulation_steps*50

    span_model, type_model, optimizer_span, scheduler_span, \
    optimizer_type, scheduler_type = initialize(args, tokenizer, t_total, span_num_labels, type_num_labels_src, type_num_labels_tgt)

    logger.info("***** Running training *****")
    logger.info("  Num examples of src = %d", len(train_dataset_src))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size_src)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size_src
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev, test = [0, 0, 0], [0, 0, 0]
    best_dev_bio, test_bio = [0, 0, 0], [0, 0, 0]
    devs = []
    tests = []

    loss_funct = MSELoss()

    iterator = iter(cycle(train_dataloader))

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader_src, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch_src in enumerate(epoch_iterator):
            span_model.train()
            type_model.train()
            batch_src = tuple(t.to(args.device) for t in batch_src)
            batch = next(iterator)
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {"input_ids": batch_src[0], "attention_mask": batch_src[1], "labels_bio": batch_src[2], "tgt": False, "reduction": "none"}
            outputs_span_src = span_model(**inputs)
            inputs = {"input_ids": batch_src[0], "attention_mask": batch_src[1], "labels_type": batch_src[3], "logits_bio": outputs_span_src[2], "tgt": False, "reduction": "none"}
            outputs_type_src = type_model(**inputs)
            loss1 = span_model.loss(outputs_span_src[0], outputs_type_src[1], tau=args.tau_span, eps=args.eps_span)
            loss2 = type_model.loss(outputs_type_src[0], outputs_span_src[1], tau=args.tau_type, eps=args.eps_type)
            loss6 = share_loss(outputs_span_src[4], outputs_type_src[4], loss_funct, args.L)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_bio": batch[2], "tgt": True, "reduction": "none"}
            outputs_span = span_model(**inputs)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_type": batch[3], "logits_bio": outputs_span[2], "tgt": True, "reduction": "none"}
            outputs_type = type_model(**inputs)
            loss3 = span_model.loss(outputs_span[0], outputs_type[1], tau=args.tau_span, eps=args.eps_span)
            permute_embed = span_model.adv_attack(outputs_span[4][0], loss3, mu=args.mu)
            inputs = {"inputs_embeds": permute_embed, "attention_mask": batch[1], "labels_bio": batch[2], "tgt": True, "reduction": "mean"}
            outputs_span_ = span_model(**inputs)
            loss31 = outputs_span_[0] 
            loss4 = type_model.loss(outputs_type[0], outputs_span[1], tau=args.tau_type, eps=args.eps_type)
            permute_embed = type_model.adv_attack(outputs_type[4][0], loss4, mu=args.mu)
            inputs = {"inputs_embeds": permute_embed, "attention_mask": batch[1], "labels_type": batch[3], "tgt": True, "reduction": "mean"}
            outputs_type_ = type_model(**inputs)
            loss41 = outputs_type_[0] 
            loss7 = share_loss(outputs_span[4], outputs_type[4], loss_funct, args.L)

            loss5 = type_model.mix_up(outputs_type_src[3], outputs_type[3], batch_src[3], batch[3], args.alpha, args.beta)

            loss = loss1+loss2+loss3+loss4+0.1*loss5+0.1*(loss6+loss7)+0.1*(loss31+loss41) # ALL

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step+1)%args.gradient_accumulation_steps == 0:
                optimizer_span.step()
                optimizer_type.step()
                scheduler_span.step()  # Update learning rate schedule
                scheduler_type.step()
                span_model.zero_grad()
                type_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step%args.logging_steps == 0:
                    # Log metrics
                    # iters.append(global_step)
                    if args.evaluate_during_training:
                        logger.info("***** training loss : %.4f*****", loss.item())
                        best_dev, test, best_dev_bio, test_bio, _ = validation(args, span_model, type_model, tokenizer, \
                            id_to_label_span, pad_token_label_id, best_dev, test, best_dev_bio, test_bio,\
                            global_step, t_total, epoch, devs, tests)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    results = (best_dev, best_test)

    return results

def main():
    args = config()
    args.do_train = args.do_train.lower()
    # args.do_test = args.do_test.lower()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", "%m/%d/%Y %H:%M:%S")
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    id_to_label_span, id_to_label_type_tgt, id_to_label_type_src = get_labels(args.data_dir, args.src_dataset, args.dataset)
    # num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = MODEL_CLASSES["span"][2].from_pretrained(
        args.tokenizer_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Loss = CycleConsistencyLoss(non_entity_id, args.device)

    # Training
    if args.do_train == "true":
        train_dataset_src, train_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train")
        best_results = train(args, train_dataset_src, train_dataset, \
            id_to_label_span, id_to_label_type_src, id_to_label_type_tgt, tokenizer, pad_token_label_id)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

if __name__ == "__main__":
    main()
