# gr_analysis.py
# plain gr with title or prefix as identifier, same as mvdr
# without contrastive loss (wo negative sampling)

import math
import os
import time
import ujson
import torch
import string
import pickle
import random
import marisa_trie
import numpy as np
from tqdm import tqdm
from seal import FMIndex
from seal import SEALSearcher
from eval import eval_result
from torch.optim import AdamW
from accelerate import Accelerator
from collections import defaultdict
from argparse import ArgumentParser
from torch.utils.data import Dataset
from torch.multiprocessing import Pool

from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
)

from utils import (
    Timer,
    clean,
    do_span,
    zipstar,
    file_tqdm,
    load_qrel,
    load_triples,
    load_queries,
    print_message,
    nltk_stopwords,
    load_candidates,
    save_checkpoint,
    load_checkpoint,
    load_collection,
    create_directory,
    get_checkpoint_id,
    exist_file_with_prefix,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QrelDataset(Dataset):
    def __init__(
        self, args, qrels=None, queries=None, collection=None, candidates=None, expand=True
    ):
        super().__init__()

        self.tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)

        self.query_maxlen = args.query_maxlen
        self.doc_maxlen = args.doc_maxlen

        self.eos_token_id = self.tok.eos_token_id

        self.queries = queries or load_queries(args.queries)
        self.qrel = qrels or load_qrel(args.qrels)

        if args.debug:
            self.qrel = self.qrel[:100]

        if collection is not None:
            self.collection = collection
        else:
            self.collection = load_collection(args.collection)

            self.collection = clean(
                self.collection,
                rm_punctuation=args.rm_punctuation,
                rm_stopwords=args.rm_stopwords,
                lower=args.lower,
                data_name=args.data_name,
            )

        self.candidates = candidates or load_candidates(args.candidates)

        if expand:
            self._expand_candidates()

    def _expand_candidates(self):
        qrels = []
        for qid, pid_list in self.candidates.items():
            for pid in pid_list:
                qrels.append((qid, pid))

        self.qrel = qrels

    def __len__(self):
        return len(self.qrel)

    def __getitem__(self, index):
        qrel = self.qrel[index]
        qid, pid = qrel
        query = self.queries[qid]
        passage = self.collection[pid]

        return qid, pid, query, passage

    def collate_fn(self, data):
        qids, pids, queries, docs = zipstar(data)

        self.tok: T5TokenizerFast

        queries_tok = self.tok(
            queries,
            max_length=self.query_maxlen,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        assert queries_tok["input_ids"].size(0) == len(qids)

        # queries_tok["input_ids"][queries_tok["input_ids"] == 0] = self.mask_token_id

        doc_tok = self.tok(
            docs,
            max_length=self.doc_maxlen,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        doc_tok["input_ids"][doc_tok["input_ids"] == 0] = -100

        return (
            qids,
            pids,
            (queries_tok["input_ids"], queries_tok["attention_mask"]),
            (doc_tok["input_ids"], doc_tok["attention_mask"]),
        )


def mean_of_list(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)

def alignment_case(args, qrels=None, queries=None, collection=None, candidates=None):
    t5 = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
    )
    tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)
    load_checkpoint(args.checkpoint, model=t5)

    t5.to(DEVICE)
    t5.eval()

    dataset = QrelDataset(
        args, qrels=qrels, queries=queries, collection=collection, candidates=candidates, expand=False
    )
    print(f"len(dataset)={len(dataset)}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    with torch.no_grad():
        resource = []
        for batch in tqdm(dataloader, desc="doing job"):
            qids, pids, queries, passages = batch
            queries = (queries[0].to(DEVICE), queries[1].to(DEVICE))
            passages = (passages[0].to(DEVICE), passages[1].to(DEVICE))

            batch_size = queries[0].size(0)

            input_dict = {
                "input_ids": queries[0],
                "attention_mask": queries[1],
                "labels": passages[0],
                "output_attentions": True,
                "return_dict": True,
            }

            outputs = t5(**input_dict)

            head_averaged_attention = outputs.cross_attentions[-1].mean(dim=1)

            assert head_averaged_attention.size() == (
                batch_size,
                passages[0].size(1),
                queries[0].size(1),
            )

            scores = head_averaged_attention

            query_ids = queries[0]
            passage_ids = passages[0]

            D_Q_is_match = passage_ids[:, :, None] == query_ids[:, None, :]
            D_padding = passage_ids == -100
            Q_padding = query_ids == 0
            D_Q_is_match = D_Q_is_match & (~D_padding[:, :, None]) & (~Q_padding[:, None, :])

            assert D_Q_is_match.size() == (batch_size, passage_ids.size(1), query_ids.size(1))

            match_score = scores * D_Q_is_match
            match_rate_per_sample = match_score.sum((1, 2))
            assert match_rate_per_sample.size() == (batch_size,)

            select_topk_sample_idx = match_rate_per_sample.argsort(descending=True)[-1:]

            scores = scores.cpu()
            match_rate_per_sample = match_rate_per_sample.cpu()
            query_ids = query_ids.cpu()
            passage_ids = passage_ids.cpu()

            for idx in select_topk_sample_idx:
                query_tok_ids = query_ids[idx]
                passage_tok_ids = passage_ids[idx]
                _scores = scores[idx]
                # remove paddings, resize _scores, query_tok_ids, passage_tok_ids
                _scores = _scores[:, query_tok_ids != 0]
                _scores = _scores[passage_tok_ids != -100, :]
                query_tok_ids = query_tok_ids[query_tok_ids != 0]
                passage_tok_ids = passage_tok_ids[passage_tok_ids != -100]
                resource.append(
                    {
                        "query_tok_ids": query_tok_ids,
                        "passage_tok_ids": passage_tok_ids,
                        "doc2query_attention": _scores,
                        "match_rate": match_rate_per_sample[idx],
                    }
                )
        
    sorted_resource = sorted(resource, key=lambda x: x["match_rate"], reverse=True)
    result_path = os.path.join(args.output_dir, f"{args.data_name}.alignment_case.pt")
    with open(result_path, "wb") as f:
        # pickle.dump(sorted_resource, f)
        torch.save(sorted_resource, f)

    plot_dir = os.path.join(args.output_dir, "alignment_case")
    os.makedirs(plot_dir, exist_ok=True)

    import seaborn as sns
    import matplotlib.pyplot as plt

    for idx, sample in enumerate(sorted_resource[:10]):
        query_tok_ids = sample["query_tok_ids"]
        passage_tok_ids = sample["passage_tok_ids"]
        doc2query_attention = sample["doc2query_attention"]

        query_tok = tok.convert_ids_to_tokens(query_tok_ids)
        passage_tok = tok.convert_ids_to_tokens(passage_tok_ids)

        for i in range(len(query_tok)):
            query_tok[i] = query_tok[i].replace("▁", "")
        for i in range(len(passage_tok)):
            passage_tok[i] = passage_tok[i].replace("▁", "")

        print("doc2query_attention.size()", doc2query_attention.size())
        attention = np.array(doc2query_attention)
        print("attention.shape", attention.shape)
        attention = attention.reshape((attention.shape[0], attention.shape[1]))

        # Set seaborn style
        sns.set(style="whitegrid", font_scale=1.5)  # Adjusted font_scale for better visibility

        # Use LaTeX for text rendering
        # plt.rc('text', usetex=True)

        # Create subplots with specified figsize for two columns
        fig, ax = plt.subplots(figsize=(4, 4))  # Adjusted figsize for better aspect ratio
        
        # Plot the heatmap
        sns.heatmap(
            attention,
            # annot=True,
            ax=ax,
            fmt=".2f",  # Display numbers with two decimal places
            vmin=0,
            vmax=1,
            cmap="YlGnBu",  # Changed to a more standard color map
            xticklabels=passage_tok,
            yticklabels=query_tok,
            cbar_kws={'label': 'Attention Score'},  # Uncommented for color bar
        )

        # Set axis labels using LaTeX
        ax.set_xlabel('Passage')
        ax.set_ylabel('Query')

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        # Save the figure with higher resolution
        plt.savefig(os.path.join(plot_dir, f"{idx}.pdf"), bbox_inches='tight', dpi=600)

    print_message("#> done", plot_dir)


def xmatch_rate_v2(args, qrels=None, queries=None, collection=None, candidates=None):
    t5 = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
    )
    tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)
    load_checkpoint(args.checkpoint, model=t5)

    t5.to(DEVICE)
    t5.eval()

    dataset = QrelDataset(
        args, qrels=qrels, queries=queries, collection=collection, candidates=candidates, expand=not args.debug
    )
    print(f"len(dataset)={len(dataset)}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    resource_path = f"{args.data_name}.gr_analysis.resource"
    resource_path = os.path.join(args.output_dir, resource_path)
    if os.path.exists(resource_path):
        with open(resource_path, "rb") as f:
            resource = pickle.load(f)
    else:
        with torch.no_grad():
            resource = defaultdict(list)
            for batch in tqdm(dataloader, desc="doing job"):
                qids, pids, queries, passages = batch
                queries = (queries[0].to(DEVICE), queries[1].to(DEVICE))
                passages = (passages[0].to(DEVICE), passages[1].to(DEVICE))

                batch_size = queries[0].size(0)

                input_dict = {
                    "input_ids": queries[0],
                    "attention_mask": queries[1],
                    "labels": passages[0],
                    "output_attentions": True,
                    "return_dict": True,
                }

                outputs = t5(**input_dict)
                decoder_input_ids = t5._shift_right(passages[0])

                head_averaged_attention = outputs.cross_attentions[-1].mean(dim=1)

                assert head_averaged_attention.size() == (
                    batch_size,
                    passages[0].size(1),
                    queries[0].size(1),
                )

                scores = head_averaged_attention

                scores.transpose(1, 2)[queries[0] == 1] = 0
                max_scores = scores.max(2)
                assert max_scores.indices.size() == (batch_size, passages[0].size(1))

                D_Q_is_match = passages[0][:, :, None] == queries[0][:, None, :]
                assert D_Q_is_match.size() == (
                    batch_size,
                    passages[0].size(1),
                    queries[0].size(1),
                )
                D_Q_is_match_score = scores * D_Q_is_match
                assert D_Q_is_match_score.size() == (
                    batch_size,
                    passages[0].size(1),
                    queries[0].size(1),
                )

                doc_tok_ids = passages[0].cpu().tolist()
                query_tok_ids = (
                    torch.gather(queries[0], 1, max_scores.indices).cpu().tolist()
                )

                del scores, decoder_input_ids

                for idx, qid in enumerate(qids):
                    resource[qid].append(
                        {
                            "doc_tok_id": doc_tok_ids[idx],
                            "query_tok_id": query_tok_ids[idx],
                            "doc_acc_match_score": D_Q_is_match_score[idx]
                            .sum(dim=1)
                            .cpu()
                            .tolist(),
                            "query_acc_match_score": D_Q_is_match_score[idx]
                            .sum(dim=0)
                            .cpu()
                            .tolist(),
                            "doc_input_ids": passages[0][idx].cpu().tolist(),
                            "query_input_ids": queries[0][idx].cpu().tolist(),
                        }
                    )

        with open(resource_path, "wb") as f:
            pickle.dump(resource, f)

    skip_token_id_set = set(tok.all_special_ids)
    skip_token_set = set(tok.all_special_tokens)
    skip_token_set.update(["", " ", "\n", "\t", "\r", "\r\n"])
    skip_token_set.update(nltk_stopwords)
    skip_token_set.update(string.punctuation)

    def should_skip(token_id):
        if token_id < 0 or token_id in skip_token_id_set:
            return True
        token = tok.decode(token_id, skip_special_tokens=False)
        if token in skip_token_set:
            return True
        return False

    match_per_tok_id = defaultdict(list)
    match_score_per_doctok_id = defaultdict(list)
    match_score_per_querytok_id = defaultdict(list)

    for qid, docs in tqdm(resource.items(), desc="xmatch_rate"):
        for doc in docs:
            for idx, (doc_tok_id, query_tok_id) in enumerate(
                zip(doc["doc_tok_id"], doc["query_tok_id"])
            ):
                if should_skip(doc_tok_id) or should_skip(query_tok_id):
                    continue
                match_per_tok_id[doc_tok_id].append(doc_tok_id == query_tok_id)

    for qid, docs in tqdm(resource.items(), desc="match_score"):
        for doc in docs:
            for query_tok_id, query_acc_match_score in zip(
                doc["query_tok_id"], doc["query_acc_match_score"]
            ):
                if should_skip(query_tok_id):
                    continue
                if query_acc_match_score > 1e-6:
                    match_score_per_querytok_id[query_tok_id].append(
                        query_acc_match_score
                    )

            for doc_tok_id, doc_acc_match_score in zip(
                doc["doc_tok_id"], doc["doc_acc_match_score"]
            ):
                if should_skip(doc_tok_id):
                    continue
                if doc_acc_match_score > 1e-6:
                    match_score_per_doctok_id[doc_tok_id].append(doc_acc_match_score)

    collection_tokenize_path = f"{args.data_name}.collection.tokenize"
    collection_tokenize_path = os.path.join(args.output_dir, collection_tokenize_path)
    if os.path.exists(collection_tokenize_path):
        with open(collection_tokenize_path, "rb") as f:
            term_df = pickle.load(f)
    else:
        term_df = {}
        with Timer("clean and merge collections"):
            if args.debug:
                iter = list(collection.values())[:100]
            else:
                iter = collection.values()
            for psg in iter:
                tok_ids = tok.encode(psg, add_special_tokens=False)
                for tok_id in set(tok_ids):
                    term_df[tok_id] = term_df.get(tok_id, 0) + 1
        with open(collection_tokenize_path, "wb") as f:
            pickle.dump(term_df, f)

    info_per_tok_id = {}
    info_per_doc_id = {}
    info_per_query_id = {}

    for tok_id in match_per_tok_id:
        df = term_df.get(tok_id, 0)
        idf = math.log(1 + (len(collection) - df + 0.5) / (df + 0.5))
        info_per_tok_id[tok_id] = {
            "xmatch_rate": mean_of_list(match_per_tok_id[tok_id]),
            "idf": idf,
            "token": tok.decode(tok_id),
        }

    for doc_tok_id in match_score_per_doctok_id:
        df = term_df.get(doc_tok_id, 0)
        idf = math.log(1 + (len(collection) - df + 0.5) / (df + 0.5))
        info_per_doc_id[doc_tok_id] = {
            "xmatch_rate": mean_of_list(match_score_per_doctok_id[doc_tok_id]),
            "idf": idf,
            "token": tok.decode(doc_tok_id),
        }

    for query_tok_id in match_score_per_querytok_id:
        df = term_df.get(query_tok_id, 0)
        idf = math.log(1 + (len(collection) - df + 0.5) / (df + 0.5))
        info_per_query_id[query_tok_id] = {
            "xmatch_rate": mean_of_list(match_score_per_querytok_id[query_tok_id]),
            "idf": idf,
            "token": tok.decode(query_tok_id),
        }

    result_path = os.path.join(
        args.output_dir, f"{args.data_name}.info_per_tok_id.json"
    )
    with open(result_path, "w") as f:
        ujson.dump(info_per_tok_id, f, indent=2)

    result_path = os.path.join(
        args.output_dir, f"{args.data_name}.info_per_doc_id.json"
    )
    with open(result_path, "w") as f:
        ujson.dump(info_per_doc_id, f, indent=2)

    result_path = os.path.join(
        args.output_dir, f"{args.data_name}.info_per_query_id.json"
    )
    with open(result_path, "w") as f:
        ujson.dump(info_per_query_id, f, indent=2)

    draw_boxplot(info_per_tok_id, args.output_dir, tag="xmatch_rate")
    draw_boxplot(info_per_doc_id, args.output_dir, tag="doc_xmatch_score")
    draw_boxplot(info_per_query_id, args.output_dir, tag="query_xmatch_score")


def draw_boxplot(info_per_tok_id, output_dir, tag="xmatch_rate"):
    """
    xmatch rate vs. idf, bin box plot
    idf is splitted into 10 bins, [0, 2), [2, 4), ..., [18, 20)
    hide outliers
    """
    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(
        style="whitegrid", rc={"grid.linewidth": 0.1, "font.family": "Source Serif Pro"}
    )
    sns.set_context("paper")
    color = sns.color_palette("Set2", 6)

    os.makedirs(output_dir, exist_ok=True)

    xmatch_rate = []
    idf = []
    for tok_id, info in info_per_tok_id.items():
        xmatch_rate.append(info["xmatch_rate"])
        idf.append(info["idf"])

    idf_bins = [0, 2, 4, 6, 8, 15]
    idf_labels = [f"[{idf_bins[i]}, {idf_bins[i+1]})" for i in range(len(idf_bins) - 1)]
    idf_bins = np.array(idf_bins)
    idf = np.array(idf)
    idf_bin_idx = np.digitize(idf, idf_bins) - 1

    xmatch_rate = np.array(xmatch_rate)

    fig, ax = plt.subplots(figsize=(3.1, 3))
    sns.boxplot(
        x=idf_bin_idx,
        y=xmatch_rate,
        ax=ax,
        showfliers=False,
        showmeans=True,
        linewidth=1.5,
        color="#1971C2",
        width=0.5,
        meanprops={"markersize": "3"},
    )
    ax.set_xticklabels(idf_labels)
    ax.set_xlabel("token idf")
    ax.set_ylabel("exact match rate")
    # ax.set_title("exact match rate of MVDR vs. token idf")
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{tag}_xmatch_vs_idf.pdf"),
        dpi=600,
        bbox_inches="tight",
    )


def xmatch_rate(args, qrels=None, queries=None, collection=None, candidates=None):
    t5 = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
    )
    tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)
    load_checkpoint(args.checkpoint, model=t5)

    t5.to(DEVICE)
    t5.eval()

    dataset = QrelDataset(
        args, qrels=qrels, queries=queries, collection=collection, candidates=candidates
    )
    print(f"len(dataset)={len(dataset)}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    resource_path = f"{args.data_name}.gr_analysis.resource"
    resource_path = os.path.join(args.output_dir, resource_path)
    if os.path.exists(resource_path):
        with open(resource_path, "rb") as f:
            resource = pickle.load(f)
    else:
        with torch.no_grad():
            resource = defaultdict(list)
            for batch in tqdm(dataloader, desc="xmatch_rate"):
                qids, pids, queries, passages = batch
                queries = (queries[0].to(DEVICE), queries[1].to(DEVICE))
                passages = (passages[0].to(DEVICE), passages[1].to(DEVICE))

                batch_size = queries[0].size(0)

                input_dict = {
                    "input_ids": queries[0],
                    "attention_mask": queries[1],
                    "labels": passages[0],
                    "output_attentions": True,
                    "return_dict": True,
                }

                outputs = t5(**input_dict)
                decoder_input_ids = t5._shift_right(passages[0])

                head_averaged_attention = outputs.cross_attentions[-1].mean(dim=1)

                assert head_averaged_attention.size() == (
                    batch_size,
                    passages[0].size(1),
                    queries[0].size(1),
                )

                scores = head_averaged_attention
                scores = scores.max(2)

                assert scores.indices.size() == (batch_size, passages[0].size(1))

                # doc tok id -> query tok id
                selected_query_tok_id = torch.gather(queries[0], 1, scores.indices)
                assert selected_query_tok_id.size() == (batch_size, passages[0].size(1))

                score_value = scores.values.cpu().numpy()
                doc_tok_ids = passages[0].cpu().numpy()
                aligned_query_tok_ids = selected_query_tok_id.cpu().numpy()

                del scores, decoder_input_ids, selected_query_tok_id

                for idx, (qid, pid) in enumerate(zip(qids, pids)):
                    resource[qid].append(
                        {
                            "pid": pid,
                            "score": score_value[idx],
                            "doc_tok_ids": doc_tok_ids[idx],
                            "aligned_query_tok_ids": aligned_query_tok_ids[idx],
                            "query_ids": queries[0][idx].cpu().numpy(),
                            "passage_ids": passages[0][idx].cpu().numpy(),
                            "doc2query_attention": outputs.cross_attentions[-1][idx]
                            .cpu()
                            .numpy(),
                        }
                    )

        with open(resource_path, "wb") as f:
            pickle.dump(resource, f)

    skip_token_id_set = set(tok.all_special_ids)
    skip_token_set = set(tok.all_special_tokens)
    skip_token_set.update(["", " ", "\n", "\t", "\r", "\r\n"])
    skip_token_set.update(nltk_stopwords)
    skip_token_set.update(string.punctuation)

    # expand attention score
    n_heads = 12
    expanded_match = defaultdict(list)
    matched_score_per_tok_id = defaultdict(list)

    for qid, docs in tqdm(resource.items()):
        for doc in docs:
            for idx, doc_tok_id in enumerate(doc["doc_tok_ids"]):
                if doc_tok_id < 0 or doc_tok_id in skip_token_id_set:
                    continue
                doc_tok = tok.decode(doc_tok_id, skip_special_tokens=False)
                if doc_tok in skip_token_set:
                    continue
                for j, query_tok_id in enumerate(doc["query_ids"]):
                    if query_tok_id == doc_tok_id:
                        matched_score_per_tok_id[doc_tok_id].extend(
                            doc["doc2query_attention"][:, idx, j]
                        )

    xmatch_rate_per_tok_id = defaultdict(float)

    for doc_tok_id in matched_score_per_tok_id:
        xmatch_rate_per_tok_id[doc_tok_id] = mean_of_list(
            matched_score_per_tok_id[doc_tok_id]
        )

    mean_xmatch_rate = mean_of_list(list(xmatch_rate_per_tok_id.values()))
    print_message(f"#> mean_xmatch_rate: {mean_xmatch_rate}")

    term_df = {}
    for psg in collection.values():
        psg_tok_id = tok.encode(psg, add_special_tokens=False)
        for tok_id in set(psg_tok_id):
            term_df[tok_id] = term_df.get(tok_id, 0) + 1

    info_per_tok_id = {}
    for tok_id in xmatch_rate_per_tok_id:
        df = term_df.get(tok_id, 0)
        idf = math.log(1 + (len(collection) - df + 0.5) / (df + 0.5))

        info_per_tok_id[tok_id] = {
            "xmatch_rate": xmatch_rate_per_tok_id[tok_id],
            "mean_delta": 0,  # mean_delta_per_tok_id[tok_id],
            "idf": idf,
            "token": tok.decode(tok_id),
        }

    result_path = os.path.join(args.output_dir, f"{args.data_name}.gr_analysis.json")
    with open(result_path, "w") as f:
        ujson.dump(info_per_tok_id, f, indent=2)

    draw_xmatch_rate_boxplot(info_per_tok_id, args.output_dir)


def draw_xmatch_rate_boxplot(info_per_tok_id, output_dir):
    """
    xmatch rate vs. idf, bin box plot
    idf is splitted into 10 bins, [0, 2), [2, 4), ..., [18, 20)
    hide outliers
    """
    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(
        style="whitegrid", rc={"grid.linewidth": 0.1, "font.family": "Source Serif Pro"}
    )
    sns.set_context("paper")
    color = sns.color_palette("Set2", 6)

    os.makedirs(output_dir, exist_ok=True)

    xmatch_rate = []
    mean_delta = []
    idf = []
    for tok_id, info in info_per_tok_id.items():
        xmatch_rate.append(info["xmatch_rate"])
        mean_delta.append(info["mean_delta"])
        idf.append(info["idf"])

    idf_bins = [0, 2, 4, 6, 8, 15]
    idf_labels = [f"[{idf_bins[i]}, {idf_bins[i+1]})" for i in range(len(idf_bins) - 1)]
    idf_bins = np.array(idf_bins)
    idf = np.array(idf)
    idf_bin_idx = np.digitize(idf, idf_bins) - 1

    xmatch_rate = np.array(xmatch_rate)
    mean_delta = np.array(mean_delta)

    fig, ax = plt.subplots(figsize=(3.1, 3))
    sns.boxplot(
        x=idf_bin_idx,
        y=xmatch_rate,
        ax=ax,
        showfliers=False,
        showmeans=True,
        linewidth=1.5,
        color="#1971C2",
        width=0.5,
        meanprops={"markersize": "3"},
    )
    ax.set_xticklabels(idf_labels)
    ax.set_xlabel("token idf")
    ax.set_ylabel("exact match rate")
    # ax.set_title("exact match rate of MVDR vs. token idf")
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "xmatch_rate_vs_idf.pdf"), dpi=600, bbox_inches="tight"
    )

    fig, ax = plt.subplots(figsize=(3.1, 3))
    # sns.boxplot(x=idf_bin_idx, y=mean_delta, ax=ax, showfliers=False)
    sns.boxplot(
        x=idf_bin_idx,
        y=mean_delta,
        ax=ax,
        showfliers=False,
        showmeans=True,
        linewidth=1.5,
        color="#1971C2",
        width=0.5,
        meanprops={"markersize": "3"},
    )
    ax.set_xticklabels(idf_labels)
    ax.set_xlabel("idf")
    ax.set_ylabel("mean delta")
    ax.set_title("mean delta vs. idf")
    fig.savefig(
        os.path.join(output_dir, "mean_delta_vs_idf.pdf"), dpi=600, bbox_inches="tight"
    )


def parse_args():
    parser = ArgumentParser("t5 training")

    parser.add_argument("--project", type=str, default="t5-Training")
    parser.add_argument(
        "--task",
        type=str,
        default="xmatch_rate_v2",
        choices=["xmatch_rate", "case_alignment", "sparsity", "rank", "xmatch_rate_v2"],
    )

    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--qrels", type=str, default=None)
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--candidates", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="corpus")

    parser.add_argument("--query_maxlen", type=int, default=32)
    parser.add_argument("--doc_maxlen", type=int, default=128)
    parser.add_argument("--rm_stopwords", action="store_true")
    parser.add_argument("--rm_punctuation", action="store_true")
    parser.add_argument("--lower", action="store_true")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--with_track", action="store_true")
    parser.add_argument("--use_training_progress_bar", action="store_true")

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # args check
    assert os.path.exists(args.model_name_or_path)
    assert os.path.exists(args.queries)
    assert os.path.exists(args.collection)

    assert "t5" in args.model_name_or_path, "only support t5 model"

    return args


if __name__ == "__main__":
    args = parse_args()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    os.environ["PYTHONHASHSEED"] = str(12345)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(12345)
        torch.cuda.manual_seed(12345)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    json_args = ujson.dumps(args.__dict__, indent=4)
    print_message("#> Arguments:", json_args)

    create_directory(args.output_dir)

    collection = load_collection(args.collection)
    queries = load_queries(args.queries)
    qrels = load_qrel(args.qrels)
    candidates = load_candidates(args.candidates)

    collection = clean(
        collection,
        rm_stopwords=args.rm_stopwords,
        rm_punctuation=args.rm_punctuation,
        lower=args.lower,
        data_name=args.data_name,
    )

    if args.task == "xmatch_rate":
        xmatch_rate(
            args,
            qrels=qrels,
            queries=queries,
            collection=collection,
            candidates=candidates,
        )
    elif args.task == "xmatch_rate_v2":
        xmatch_rate_v2(
            args,
            qrels=qrels,
            queries=queries,
            collection=collection,
            candidates=candidates,
        )
    elif args.task == "case_alignment":
        alignment_case(
            args,
            qrels=qrels,
            queries=queries,
            collection=collection,
            candidates=candidates,
        )
    else:
        raise NotImplementedError
