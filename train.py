import json
import logging
import math
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)


from parsers import train_args
from utils import (
    PreprocessOfSummarizationTrain,
    postprocess_text,
    save_log,
)

# torch.cuda.is_available()
# from tw_rouge import get_rouge  # will cause run out of memory problem

logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
UID = str(uuid.uuid1())


def main(args):
    accelerator = Accelerator()
    logger.info(accelerator.state, main_process_only=False)

    set_seed(args.seed)

    # dev_files = ["./data/train.jsonl", "./data/public.jsonl"]
    raw_datasets = load_dataset(
        "summary_generator.py",
        name="Summary Dataset",
        cache_dir="./cache",
        jsonl_files=[args.train_file, args.validation_file],
        split_names=["train", "eval"],
    )

    # Config Tokenizer and Model
    if args.resume_ckpt:
        print(f"resume model from {args.resume_ckpt}")
        config = AutoConfig.from_pretrained(args.resume_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(args.resume_ckpt, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.resume_ckpt, config=config)
        model.resize_token_embeddings(len(tokenizer))
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)  #  use_fast=False
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names
    preprocess_function = PreprocessOfSummarizationTrain(tokenizer=tokenizer)

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Metric
    metric = load_metric("rouge")
    # metric = get_rouge

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Use fp 16: {accelerator.use_fp16}")

    completed_steps = 0
    starting_epoch = 0
    log = {
        "train_loss": [],
        "eval_rouge1": [],
        "eval_rouge2": [],
        "eval_rougeL": [],
    }
    result = {}

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0

        epoch_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=100)
        for step, batch in epoch_pbar:
            outputs = model(**batch)
            loss = outputs.loss

            # We keep track of the loss at each epoch
            total_loss += loss.item()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                # Set Progress Bar
                epoch_pbar.set_description(f"Epoch[{epoch + 1}/{args.num_train_epochs}] @step {step:6d}")
                epoch_pbar.set_postfix(loss=total_loss / (step + 1))

            if completed_steps >= args.max_train_steps:
                break

            if step == 20 and args.dev:
                break

        # Evaluation per epoch
        model.eval()
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
        }

        all_pred = []
        all_label = []

        epoch_pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), ncols=100)
        for step, batch in epoch_pbar:
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                all_pred += decoded_preds
                all_label += decoded_labels

                # print(f"all_pred:{len(all_pred)}, all_label:{len(all_label)}")

                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )

                if step == 20 and args.dev:
                    break

            # Set Progress Bar
            epoch_pbar.update(1)

        print(f"Decode label: {all_label[:3]}")
        print(f"Decode pred: {all_pred[:3]}")

        result = metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"Valid Metric: {result}")

        log["train_loss"].append(total_loss / len(train_dataloader))
        log["eval_rouge1"].append(result["rouge1"])
        log["eval_rouge2"].append(result["rouge2"])
        log["eval_rougeL"].append(result["rougeL"])

    # Save checkpoint
    if args.ckpt_dir is not None and not args.dev:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.ckpt_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.ckpt_dir)

        with open(os.path.join(args.ckpt_dir, "all_results.json"), "w") as f:
            json.dump(
                {
                    "eval_rouge1": result["rouge1"],
                    "eval_rouge2": result["rouge2"],
                    "eval_rougeL": result["rougeL"],
                    "eval_rougeLsum": result["rougeLsum"],
                },
                f,
            )

        save_log(log, os.path.join(args.ckpt_dir, "log.json"))


if __name__ == "__main__":
    args = train_args()
    args.ckpt_dir = Path(f"{args.ckpt_dir}/{UID[:8]}")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
