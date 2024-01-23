# eval.py

import os
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from utils import create_directory


def load_results(result_path):
    # result_path = os.path.join(file_path, "result.json")
    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)
    return result_data


def sort_results_by_score(results):
    sorted_results = [
        sorted(pid_score_list, key=lambda x: x[1], reverse=True)
        for pid_score_list in results
    ]

    return sorted_results


def eval_all(predict, label, at=[1, 5, 10, 50, 100, 1000]):
    log_dict = {}
    for k in at:
        log_dict.update(eval_recall(predict, label, at=k))
        log_dict.update(eval_mrr(predict, label, at=k))
        log_dict.update(eval_ndcg(predict, label, at=k))
    return log_dict


def base_it(predict, label, at, score_func):
    assert len(predict) == len(label)
    scores = []
    for pred, lbs in zip(predict, label):
        pred = pred if not isinstance(pred, list) else [pid for pid, *_ in pred]
        best_score = 0.0
        if not isinstance(lbs, list):
            lbs = [lbs]
        for lb in lbs:
            if isinstance(lb, list):
                lb = lb[0]
            rank = pred[:at].index(lb) + 1 if lb in pred[:at] else 0
            cur_score = score_func(rank)
            best_score = max(best_score, cur_score)
        scores.append(best_score)
    return scores


def eval_recall(predict, label, at=10):
    scores = base_it(predict, label, at, lambda rank: int(rank != 0))
    return {f"Recall@{at}": sum(scores) / len(scores)}


def eval_mrr(predict, label, at=10):
    scores = base_it(predict, label, at, lambda rank: 1 / rank if rank != 0 else 0)
    return {f"MRR@{at}": sum(scores) / len(scores)}


def eval_ndcg(predict, label, at=10):
    scores = base_it(
        predict, label, at, lambda rank: 1 / np.log2(rank + 1) if rank != 0 else 0
    )
    return {f"ndcg@{at}": sum(scores) / len(scores)}


def eval_result(results, at, output_path=None, model_ckpt_path=None):
    """
    results: list of dict
    """
    if model_ckpt_path is not None:
        ckpt_id = model_ckpt_path.split("/")[-1].split(".")[0]
        metric_file_name = f"metrics_{ckpt_id}.json"
    else:
        metric_file_name = "metrics.json"

    keys = results[0].keys()
    labels = []
    predict_list = []

    for result in results:
        predict_list.append(result["ranking_result"])
        if "target_passage" in keys:
            labels.append(result["target_passage"])
        else:
            labels.append(result["target_pid"])

    sorted_results = sort_results_by_score(predict_list)
    print(sorted_results[:5])

    result_metrics = eval_all(sorted_results, labels, at=at)

    if output_path is not None:
        create_directory(output_path)
        metric_path = os.path.join(output_path, metric_file_name)
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(result_metrics, f, indent=2)

    return result_metrics


if __name__ == "__main__":

    def parse_args():
        parser = ArgumentParser()
        parser.add_argument("--result_path", type=str, required=True)
        parser.add_argument("--output_path", type=str, default="output")
        parser.add_argument("--at", nargs="*", type=int, default=[1, 5, 10, 50])

        args = parser.parse_args()

        print(args.__dict__)

        return args

    args = parse_args()

    results = load_results(args.result_path)

    result_metrics = eval_result(results, args.at)

    print(result_metrics)

    metric_path = os.path.join(args.output_path, "metrics.json")
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(result_metrics, f, indent=2)
