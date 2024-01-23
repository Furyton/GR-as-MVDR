# mvdr.py
# ColT5.v1
# add marker token for query and document

from collections import defaultdict
import os
import re
import string
import time
import h5py
import math
import ujson
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from itertools import accumulate
from accelerate import Accelerator
from pyserini.search import SimpleSearcher
from pyserini.index.lucene import IndexReader
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertModel,
    T5EncoderModel,
    AutoTokenizer,
    T5TokenizerFast,
)

from faiss_index import FaissIndex, FaissRetrieveIndex
from utils import (
    Timer,
    clean,
    zipstar,
    load_qrel,
    load_queries,
    load_triples,
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


class ColT5(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path="t5-base",
        query_maxlen=32,
        doc_maxlen=128,
        dim=128,
        mask_punctuation=False,
        similarity_metric="cosine",
    ):
        super(ColT5, self).__init__()

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            raise NotImplementedError

            self.tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
            self.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [
                    symbol,
                    self.tokenizer.encode(symbol, add_special_tokens=False)[0],
                ]
            }

        if "t5" in model_name_or_path:
            self.encoder = T5EncoderModel.from_pretrained(model_name_or_path)
        else:
            self.encoder = BertModel.from_pretrained(model_name_or_path)
        self.linear = torch.nn.Linear(self.encoder.config.hidden_size, dim, bias=False)

        self.linear.weight.data.normal_(
            mean=0.0, std=(self.encoder.config.hidden_size) ** -0.5
        )

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims="return_mask")

        # return self.score(self.query(*Q), self.doc(*D))
        return self.score(Q, D, D_mask)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.encoder(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, "return_mask"]

        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.encoder(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        # mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        mask = attention_mask.unsqueeze(2).float()
        assert (
            attention_mask.shape == D.shape[:2]
        ), f"{attention_mask.shape} != {D.shape[:2]}"
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        # TODO
        # D = D.half()

        if keep_dims is False:
            # TODO: float16 -> float32
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        elif keep_dims == "return_mask":
            return D, mask.bool()

        return D

    def score(self, Q, D, D_mask):
        if self.similarity_metric == "cosine":
            scores = D @ Q.permute(0, 2, 1)
            D_padding = ~D_mask.view(scores.size(0), scores.size(1)).bool()
            scores[D_padding] = float("-inf")
            scores = scores.max(1).values.sum(-1)
            # return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
            return scores

        assert self.similarity_metric == "l2"
        scores = -1.0 * ((D.unsqueeze(2) - Q.unsqueeze(1)) ** 2).sum(-1)
        D_padding = ~D_mask.view(scores.size(0), scores.size(1)).bool()
        scores[D_padding] = float("-inf")
        scores = scores.max(1).values.sum(-1)
        # return (
        #     (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1))
        #     .max(-1)
        #     .values.sum(-1)
        # )
        return scores

    def mask(self, input_ids):
        mask = [
            [(x not in self.skiplist) and (x != 0) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask


class QrelDataset(Dataset):
    def __init__(
        self, args, qrels=None, queries=None, collection=None, candidates=None
    ):
        super().__init__()

        if "t5" in args.model_name_or_path:
            self.tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)
        else:
            self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path)

        self.query_maxlen = args.query_maxlen
        self.doc_maxlen = args.doc_maxlen

        if "t5" in args.model_name_or_path:
            self.mask_token_id = self.tok.get_sentinel_token_ids()[0]
        else:
            self.mask_token_id = self.tok.mask_token_id

        self.queries = queries or load_queries(args.queries)
        self.qrels = qrels or load_qrel(args.qrels)

        if collection is not None:
            self.collection = collection
        else:
            self.collection = load_collection(args.collection)
            self.collection = clean(
                self.collection,
                rm_stopwords=args.rm_stopwords,
                rm_punctuation=args.rm_punctuation,
                lower=args.lower,
                data_name=args.data_name,
            )

        self.candidates = candidates or load_candidates(args.candidates)
        assert self.candidates is not None

        self._expand_candidates()

    def _expand_candidates(self):
        qrels = []
        for qid, pid_list in self.candidates.items():
            for pid in pid_list:
                qrels.append((qid, pid))

        self.qrels = qrels

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, index):
        qid, pid = self.qrels[index]
        query = self.queries[qid]
        doc = self.collection[pid]

        return qid, pid, query, doc

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

        queries_tok["input_ids"][queries_tok["input_ids"] == 0] = self.mask_token_id

        doc_tok = self.tok(
            docs,
            max_length=self.doc_maxlen,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

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


def xmatch_rate(args, qrels=None, queries_dict=None, collection=None, candidates=None):
    colt5 = ColT5(
        model_name_or_path=args.model_name_or_path,
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=args.dim,
        similarity_metric=args.similarity_metric,
    )
    load_checkpoint(args.checkpoint, model=colt5)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    colt5.to(DEVICE)
    colt5.eval()

    is_q2d = "q2d" if args.query2doc else "d2q"
    resource_path = f"{args.data_name}.{is_q2d}.mvdr_analysis.resource"
    resource_path = os.path.join(args.output_dir, resource_path)
    # resource_path = "mvdr_analysis.resource"
    if os.path.exists(resource_path):
        with open(resource_path, "rb") as f:
            resource = pickle.load(f)
    else:
        dataset = QrelDataset(
            args,
            queries=queries_dict,
            collection=collection,
            candidates=candidates,
            qrels=qrels,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            drop_last=False,
        )

        with torch.no_grad():
            resource = defaultdict(list)
            for batch in tqdm(dataloader, desc="xmatch_rate"):
                qids, pids, queries, docs = batch
                batch_size = len(qids)

                queries = (queries[0].to(DEVICE), queries[1].to(DEVICE))
                docs = (docs[0].to(DEVICE), docs[1].to(DEVICE))

                Q = colt5.query(*queries)
                D, D_mask = colt5.doc(*docs, keep_dims="return_mask")
                scores = D @ Q.permute(0, 2, 1)

                if args.query2doc:
                    D_padding = ~D_mask.view(scores.size(0), scores.size(1)).bool()
                    scores[D_padding] = float("-inf")
                    max_scores = scores.max(1)
                    assert max_scores.indices.size() == (batch_size, args.query_maxlen)
                else:
                    # [B, D, Q] if doc2query
                    max_scores = scores.max(2)
                    assert max_scores.indices.size() == (batch_size, args.doc_maxlen)

                assert scores.size() == (batch_size, args.doc_maxlen, args.query_maxlen)
                softmax_scores = torch.softmax(scores, dim=2)
                assert softmax_scores.size() == (
                    batch_size,
                    args.doc_maxlen,
                    args.query_maxlen,
                )

                D_Q_is_match = docs[0][:, :, None] == queries[0][:, None, :]
                assert D_Q_is_match.size() == (
                    batch_size,
                    args.doc_maxlen,
                    args.query_maxlen,
                )
                D_Q_match_score = softmax_scores * D_Q_is_match.float()
                assert D_Q_match_score.size() == (
                    batch_size,
                    args.doc_maxlen,
                    args.query_maxlen,
                )

                if args.query2doc:
                    doc_tok_ids = (
                        torch.gather(docs[0], 1, max_scores.indices).cpu().tolist()
                    )
                    query_tok_ids = queries[0].cpu().tolist()
                else:
                    doc_tok_ids = docs[0].cpu().tolist()
                    query_tok_ids = (
                        torch.gather(queries[0], 1, max_scores.indices).cpu().tolist()
                    )

                assert len(doc_tok_ids) == len(query_tok_ids)

                del Q, D, D_mask, scores

                for idx, (qid, pid) in enumerate(zip(qids, pids)):
                    resource[qid].append(
                        {
                            "doc_tok_id": doc_tok_ids[idx],
                            "query_tok_id": query_tok_ids[idx],
                            "doc_acc_match_score": D_Q_match_score[idx].sum(1).tolist(),
                            "query_acc_match_score": D_Q_match_score[idx]
                            .sum(0)
                            .tolist(),
                            "doc_input_ids": docs[0][idx].cpu().tolist(),
                            "query_input_ids": queries[0][idx].cpu().tolist(),
                        }
                    )

        with open(resource_path, "wb") as f:
            pickle.dump(resource, f)

    skip_token_id_set = set(tokenizer.all_special_ids)
    skip_token_set = set(tokenizer.all_special_tokens)
    skip_token_set.update(["", " ", "\n", "\t", "\r", "\r\n"])
    skip_token_set.update(nltk_stopwords)
    skip_token_set.update(string.punctuation)

    def should_skip(token_id):
        if token_id < 0 or token_id in skip_token_id_set:
            return True
        token = tokenizer.decode(token_id, skip_special_tokens=False)
        if token in skip_token_set:
            return True
        return False

    match_per_tok_id = defaultdict(list)
    match_score_per_doctok_id = defaultdict(list)
    match_score_per_querytok_id = defaultdict(list)

    for qid, docs in tqdm(resource.items(), desc="xmatch_rate"):
        for doc in docs:
            for doc_tok_id, query_tok_id in zip(doc["doc_tok_id"], doc["query_tok_id"]):
                source_tok_id = query_tok_id if args.query2doc else doc_tok_id
                if should_skip(source_tok_id):
                    continue
                match_per_tok_id[source_tok_id].append(doc_tok_id == query_tok_id)

    for qid, docs in tqdm(resource.items(), desc="match_score"):
        for doc in docs:
            for query_tok_id, query_acc_match_score in zip(
                doc["query_input_ids"], doc["query_acc_match_score"]
            ):
                if should_skip(query_tok_id):
                    continue
                if query_acc_match_score > 1e-6:
                    match_score_per_querytok_id[query_tok_id].append(
                        query_acc_match_score
                    )

            for doc_tok_id, doc_acc_match_score in zip(
                doc["doc_input_ids"], doc["doc_acc_match_score"]
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
            for psg in collection.values():
                tok_ids = tokenizer.encode(psg, add_special_tokens=False)
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
            "token": tokenizer.decode(tok_id),
        }

    for doc_tok_id in match_score_per_doctok_id:
        df = term_df.get(doc_tok_id, 0)
        idf = math.log(1 + (len(collection) - df + 0.5) / (df + 0.5))
        info_per_doc_id[doc_tok_id] = {
            "xmatch_rate": mean_of_list(match_score_per_doctok_id[doc_tok_id]),
            "idf": idf,
            "token": tokenizer.decode(doc_tok_id),
        }

    for query_tok_id in match_score_per_querytok_id:
        df = term_df.get(query_tok_id, 0)
        idf = math.log(1 + (len(collection) - df + 0.5) / (df + 0.5))
        info_per_query_id[query_tok_id] = {
            "xmatch_rate": mean_of_list(match_score_per_querytok_id[query_tok_id]),
            "idf": idf,
            "token": tokenizer.decode(query_tok_id),
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

    draw_xmatch_rate_boxplot(info_per_tok_id, args.output_dir, f"{is_q2d}_xmatch_rate")
    draw_xmatch_rate_boxplot(info_per_doc_id, args.output_dir, "doc_xmatch_score")
    draw_xmatch_rate_boxplot(info_per_query_id, args.output_dir, "query_xmatch_score")


def draw_xmatch_rate_boxplot(info_per_tok_id, output_dir, tag="xmatch_rate"):
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


def parse_args():
    parser = ArgumentParser("ColT5 / ColBERT training")

    parser.add_argument("--project", type=str, default="ColT5-Training")
    parser.add_argument(
        "--task",
        type=str,
        default="xmatch_rate",
        choices=["xmatch_rate", "case_alignment", "sparsity", "rank"],
    )

    parser.add_argument("--model_name_or_path", type=str, default="t5-base")

    parser.add_argument("--query2doc", action="store_true")
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--qrels", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--candidates", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="corpus")

    parser.add_argument("--query_maxlen", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--doc_maxlen", type=int, default=128)
    parser.add_argument("--rm_stopwords", action="store_true")
    parser.add_argument("--rm_punctuation", action="store_true")
    parser.add_argument("--lower", action="store_true")
    parser.add_argument(
        "--similarity_metric", type=str, choices=["cosine", "l2"], default="cosine"
    )
    parser.add_argument("--dim", type=int, default=128)

    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    assert os.path.exists(args.model_name_or_path)
    assert os.path.exists(args.queries)
    assert os.path.exists(args.collection)
    assert args.checkpoint is not None

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
    json_args = ujson.dumps(args.__dict__, indent=2)
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
            queries_dict=queries,
            collection=collection,
            candidates=candidates,
        )
    else:
        raise NotImplementedError
