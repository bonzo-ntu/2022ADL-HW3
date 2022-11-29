from typing import List

import datasets
import jsonlines

logger = datasets.logging.get_logger(__name__)


class SummaryConfig(datasets.BuilderConfig):
    """
    BuilderConfig for Summary dataset.
    """

    jsonl_files: List[str] = None
    split_names: List[str] = None
    split_map = {
        "train": datasets.Split.TRAIN,
        "eval": datasets.Split.VALIDATION,
        "test": datasets.Split.TEST,
    }

    def __init__(self, **kwargs):
        super(SummaryConfig, self).__init__(**kwargs)


class SummaryDataset(datasets.GeneratorBasedBuilder):
    """
    Summary dataset.
    """

    BUILDER_CONFIGS = [
        SummaryConfig(
            name="Summary Dataset",
            version=datasets.Version("1.0.0", "An initial version to load dataset from jsonlines files"),
            description="Summarization task dataset from jsonlines files",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Summary Dataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "maintext": datasets.Value("string"),
                    "title": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        split_map = {
            "train": datasets.Split.TRAIN,
            "eval": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }
        splits = [split_map[split_name] for split_name in self.config.split_names]

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "jsonl_file": jsonl_file,
                    "split_name": split_name,
                },
            )
            for split, jsonl_file, split_name in zip(splits, self.config.jsonl_files, self.config.split_names)
        ]

    def _generate_examples(self, jsonl_file, split_name):
        """
        Returns the examples in the raw (text) form.
        """
        logger.info(f"generating examples from = {jsonl_file}")

        key = 0
        with open(jsonl_file, encoding="utf-8") as f:
            jsonl = jsonlines.Reader(f)

            if split_name == "test":
                for sample in jsonl:
                    yield key, {"id": sample["id"], "maintext": sample["maintext"], "title": ""}
                    key += 1
            else:
                for sample in jsonl:
                    yield key, {"id": sample["id"], "maintext": sample["maintext"], "title": sample["title"]}
                    key += 1
