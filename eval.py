import json
from parsers import eval_args
import pandas as pd
from tw_rouge import get_rouge


def main(args):
    ref = pd.read_json(args.reference, lines=True)
    pred = pd.read_json(args.prediction, lines=True)

    # re-index then align by id
    ref.index, pred.index = ref.id.values, pred.id.values
    pred = pred.loc[ref.index]

    refs = ref.title.to_list()
    preds = pred.title.to_list()

    print(json.dumps(get_rouge(preds, refs), indent=4))


if __name__ == "__main__":
    args = eval_args()
    main(args)
