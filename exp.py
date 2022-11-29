import json
from parsers import test_args

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    set_seed,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

from utils import PreprocessOfSummarizationTrain


accelerator = Accelerator()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(1)


def main(args):
    # Config Tokenizer and Model
    # config = AutoConfig.from_pretrained(args.ckpt)
    # tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt, config=config)
    # model.resize_token_embeddings(len(tokenizer))

    print(f"resume model from {args.resume_ckpt}")
    config = AutoConfig.from_pretrained(args.resume_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.resume_ckpt, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.resume_ckpt, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # Dataset
    raw_datasets = load_dataset(
        "summary_generator.py",
        name="Summary Dataset",
        cache_dir="./cache",
        jsonl_files=[args.test_file],
        split_names=["eval"],
    )

    # TODO: make utils function to keep id column
    all_id = [features["id"] for features in raw_datasets["validation"]]
    column_names = raw_datasets["validation"].column_names

    preprocess_function = PreprocessOfSummarizationTrain(tokenizer=tokenizer)
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=["id", "maintext", "title"],  # column_names,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = processed_datasets["validation"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    gen_kwargs_set = [
        {
            "max_length": args.max_target_length,
            "num_beams": args.num_beams,
        },
        {
            "max_length": args.max_target_length,
            "num_beams": 2,
            "num_return_sequences": 1,
            "early_stopping": True,
        },
        {
            "max_length": args.max_target_length,
            "num_beams": 4,
            "num_return_sequences": 1,
            "early_stopping": True,
        },
        {
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 0,
            "temperature": 0.3,
        },
        {
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 0,
            "temperature": 0.9,
        },
        {
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 5,
        },
        {
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 20,
        },
        {
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_p": 0.75,
        },
        {
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_p": 0.95,
        },
    ]
    output_files = [
        "./experiments/result-origin.jsonl",
        "./experiments/result-beams_2.jsonl",
        "./experiments/result-beams_4.jsonl",
        "./experiments/result-temp_0.3.jsonl",
        "./experiments/result-temp_0.9.jsonl",
        "./experiments/result-top_k_5.jsonl",
        "./experiments/result-top_k_20.jsonl",
        "./experiments/result-top_p_0.75.jsonl",
        "./experiments/result-top_p_0.95.jsonl",
    ]

    # Testing
    for gen_kwargs, output_file in zip(gen_kwargs_set, output_files):
        all_pred = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather(generated_tokens)
                generated_tokens = generated_tokens.cpu().numpy()

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_pred = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                all_pred += decoded_pred

                if step == 1 and args.dev:
                    break

        print(all_pred[:3], all_id[:3])

        if not args.dev:
            # Write result to csv file
            with open(output_file, "w") as file:
                for idx, title in zip(all_id, all_pred):
                    json.dump({"title": title, "id": idx}, file)
                    file.write("\n")


if __name__ == "__main__":
    arges = test_args()
    main(arges)
