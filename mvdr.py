# mvdr.py
# ColT5.v1
# add marker token for query and document

import os
import time
import h5py
import math
import ujson
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from itertools import accumulate
from accelerate import Accelerator
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertModel,
    T5EncoderModel,
    AutoTokenizer,
    T5TokenizerFast,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
)

from eval import eval_result
from faiss_index import FaissIndex, FaissRetrieveIndex
from utils import (
    Timer,
    clean,
    zipstar,
    load_qrel,
    load_queries,
    load_triples,
    print_message,
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


class TrainDataset(Dataset):
    def __init__(self, args, triples=None, queries=None, collection=None):
        if "t5" in args.model_name_or_path:
            self.tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)
        else:
            self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.query_maxlen = args.query_maxlen
        self.doc_maxlen = args.doc_maxlen
        self.in_batch_negative = args.in_batch_negative

        if "t5" in args.model_name_or_path:
            self.mask_token_id = self.tok.get_sentinel_token_ids()[0]
        else:
            self.mask_token_id = self.tok.mask_token_id

        self.triples = triples or load_triples(args.triples)
        self.queries = queries or load_queries(args.queries)
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

        if not self.in_batch_negative:
            self._expand_triples()

    def _expand_triples(self):
        triples = []
        for qid, pos, *negs in self.triples:
            for neg in negs:
                triples.append((qid, pos, neg))

        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        if not self.in_batch_negative:
            query, pos, neg = self.triples[index]
            query, pos, neg = (
                self.queries[query],
                self.collection[pos],
                self.collection[neg],
            )

            return query, pos, neg
        else:
            query, pos, *negs = self.triples[index]
            query, pos = self.queries[query], self.collection[pos]
            return query, pos

    def collate_fn(self, batch):
        if self.in_batch_negative:
            query, pos = zip(*batch)
            query, pos = list(query), list(pos)
            n_samples = len(query)

            query_obj = self.tok(
                query,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.query_maxlen,
            )
            query_ids, query_mask = query_obj["input_ids"], query_obj["attention_mask"]
            query_ids[query_ids == 0] = self.mask_token_id

            doc_obj = self.tok(
                pos,
                padding="longest",
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.doc_maxlen,
            )
            doc_ids, doc_mask = doc_obj["input_ids"], doc_obj["attention_mask"]

            return (query_ids, query_mask), (doc_ids, doc_mask)
        else:
            query, pos, neg = zip(*batch)
            query, pos, neg = list(query), list(pos), list(neg)
            n_samples = len(query)

            # tensorize query
            # query_text = [". " + x for x in query]
            query_obj = self.tok(
                query,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.query_maxlen,
            )
            query_ids, query_mask = query_obj["input_ids"], query_obj["attention_mask"]
            # query_ids[:, 1] = self.Q_marker_token_id
            query_ids[query_ids == 0] = self.mask_token_id

            # tensorize pos and neg
            # doc_text = [". " + x for x in pos + neg]
            doc_obj = self.tok(
                pos + neg,
                padding="longest",
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.doc_maxlen,
            )
            doc_ids, doc_mask = doc_obj["input_ids"], doc_obj["attention_mask"]
            # doc_ids[:, 1] = self.D_marker_token_id
            doc_ids, doc_mask = doc_ids.view(2, n_samples, -1), doc_mask.view(
                2, n_samples, -1
            )

            maxlens = doc_mask.sum(-1).max(0).values

            # Sort by maxlens
            indices = maxlens.sort().indices
            query_ids, query_mask = query_ids[indices], query_mask[indices]
            doc_ids, doc_mask = doc_ids[:, indices], doc_mask[:, indices]

            (pos_ids, neg_ids), (pos_mask, neg_mask) = doc_ids, doc_mask

            Q = (torch.cat((query_ids, query_ids)), torch.cat((query_mask, query_mask)))
            D = (torch.cat((pos_ids, neg_ids)), torch.cat((pos_mask, neg_mask)))

            return Q, D


class DocDataset(Dataset):
    def __init__(self, args, collection=None):
        super().__init__()

        if "t5" in args.model_name_or_path:
            self.tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)
        else:
            self.tok = AutoTokenizer.from_pretrained(args.model_name_or_path)

        self.doc_maxlen = args.doc_maxlen

        if collection is not None:
            self.collections = collection
        else:
            self.collections = collection or load_collection(
                args.collection, return_dict=False
            )
            self.collections = clean(
                self.collections,
                rm_stopwords=args.rm_stopwords,
                rm_punctuation=args.rm_punctuation,
                lower=args.lower,
                is_corpus_list=True,
                data_name=args.data_name,
            )

        if type(self.collections) == dict:
            self.collections = [
                {"pid": k, "passage": v} for k, v in self.collections.items()
            ]

    def __len__(self):
        return len(self.collections)

    def __getitem__(self, index):
        doc = self.collections[index]

        return [doc["pid"]], [doc["passage"]]

    def collate_fn(self, data):
        pids, passages = zip(*data)
        pids = sum(pids, [])
        passages = sum(passages, [])

        self.tok: T5TokenizerFast

        passages_tok = self.tok(
            passages,
            max_length=self.doc_maxlen,
            padding="longest",
            truncation="longest_first",
            return_tensors="pt",
        )
        assert passages_tok["input_ids"].size(0) == len(pids)

        return pids, (passages_tok["input_ids"], passages_tok["attention_mask"])


class QrelDataset(Dataset):
    def __init__(self, args, qrels=None, queries=None):
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
        self.qrel = qrels or load_qrel(args.qrels)

    def __len__(self):
        return len(self.qrel)

    def __getitem__(self, index):
        qrel = self.qrel[index]
        qid, pid = qrel
        query = self.queries[qid]

        return qid, pid, query
        # return [query["qid"]], [query["query"], ]

    def collate_fn(self, data):
        qids, pids, queries = zipstar(data)

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

        return (
            torch.tensor(qids, dtype=torch.int),
            torch.tensor(pids, dtype=torch.int),
            (queries_tok["input_ids"], queries_tok["attention_mask"]),
        )


def train(args, collection=None, triples=None, queries=None):
    if args.with_track:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.accumsteps,
            log_with="all",
            project_dir=args.output_dir,
            # mixed_precision="fp16",
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.accumsteps,
            # mixed_precision="fp16",
        )

    colt5 = ColT5(
        model_name_or_path=args.model_name_or_path,
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        similarity_metric=args.similarity_metric,
    )

    with Timer(description="Loading dataset"):
        dataset = TrainDataset(
            args, triples=triples, queries=queries, collection=collection
        )

    accelerator.print(f"#> Loaded {len(dataset)} triples.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    tot_steps = len(dataloader) * args.n_epochs
    print_message("#> total number of steps:", tot_steps)

    if args.tot_n_ckpts > 1:
        args.save_steps = max(tot_steps // (args.tot_n_ckpts - 1) - 1, 0)
        print_message("#> new save_steps:", args.save_steps)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, colt5.parameters()), lr=args.lr, eps=1e-8
    )
    scheduler = get_constant_schedule(optimizer)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=50,
    #     num_training_steps=(len(dataloader) * args.n_epochs) // args.accumsteps,
    # )

    colt5, optimizer, dataloader, scheduler = accelerator.prepare(
        colt5, optimizer, dataloader, scheduler
    )
    # colt5 = torch.compile(colt5)

    if args.with_track:
        accelerator.init_trackers(args.project, args.__dict__)

    criterion = torch.nn.CrossEntropyLoss()
    if args.in_batch_negative:
        labels = torch.arange(args.train_batch_size, dtype=torch.long, device=DEVICE)
    else:
        labels = torch.zeros(args.train_batch_size, dtype=torch.long, device=DEVICE)

    batch_idx = torch.arange(args.train_batch_size, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    step = 0
    training_log = []
    checkpoint_path_list = []

    for epoch in range(args.n_epochs):
        accelerator.print(f"#> Epoch {epoch} started.")
        accelerator.wait_for_everyone()
        colt5.train()

        iterator = (
            tqdm(
                dataloader,
                desc="Training",
                disable=not accelerator.is_local_main_process,
                total=len(dataloader),
            )
            if args.use_training_progress_bar
            else dataloader
        )

        loss_report = []

        for idx, batch in enumerate(iterator):
            step += 1
            with accelerator.accumulate(colt5):
                query, passage = batch
                if args.in_batch_negative:
                    Q = colt5.query(*query)  # [batch_size, query_maxlen, dim]
                    D, D_mask = colt5.doc(
                        *passage, keep_dims="return_mask"
                    )  # [batch_size, doc_maxlen, dim], [batch_size, doc_maxlen, 1]

                    scores = Q.unsqueeze(1) @ D.permute(
                        0, 2, 1
                    )  # [batch_size, batch_size, query_maxlen, doc_maxlen]
                    D_padding = (
                        ~D_mask.view(1, D.size(0), 1, D.size(1))
                        .bool()
                        .expand_as(scores)
                    )
                    scores[D_padding] = float("-inf")
                    scores = scores.max(-1).values.sum(-1)
                    assert scores.size() == (Q.size(0), D.size(0))
                else:
                    scores = colt5(query, passage).view(2, -1).permute(1, 0)

                # scores = colt5(query, passage).view(2, -1).permute(1, 0)
                loss = criterion(scores, labels[: scores.size(0)])

                accelerator.backward(loss)
                if args.max_grad_norm > 0:  # and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        parameters=colt5.parameters(),
                        max_norm=args.max_grad_norm,
                        norm_type=2,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    loss = accelerator.gather(loss).mean()
                    if accelerator.is_local_main_process:
                        loss_report.append(loss.item())

                        ma_loss = np.mean(loss_report[-100:])
                        cur_loss = loss_report[-1]
                        pos_avg = scores[
                            batch_idx[: scores.size(0)], labels[: scores.size(0)]
                        ]
                        neg_avg = (scores.sum(-1) - pos_avg) / (scores.size(1) - 1)
                        pos_avg = pos_avg.mean().item()
                        neg_avg = neg_avg.mean().item()
                        diff_avg = pos_avg - neg_avg
                        # pos_avg = scores[:, 0].mean().item()
                        # neg_avg = scores[:, 1].mean().item()
                        # diff_avg = (scores[:, 0] - scores[:, 1]).mean().item()

                        if args.use_training_progress_bar:
                            iterator.set_postfix_str(
                                f"Moving Average Loss {ma_loss:.4f} | CurLoss {cur_loss:.4f}"
                            )

                        duration = time.time() - start_time
                        accelerator.print(
                            f"#> Epoch {epoch} | Step {idx} | CurLoss {cur_loss:.4f} | "
                            f"Average Loss {ma_loss:.4f} | Postive Average {pos_avg} | "
                            f"Negative Average {neg_avg} | Diff Average {diff_avg} | "
                            f"Time {duration}s | Rest {(duration / (step + 1)) * (tot_steps - step - 1) / 60:.2f}min"
                        )
                        log = {
                            "epoch": epoch,
                            "step": step,
                            "cur_loss": cur_loss,
                            "ma_loss": ma_loss,
                            "pos_avg": pos_avg,
                            "neg_avg": neg_avg,
                            "diff_avg": diff_avg,
                        }
                        if args.with_track:
                            accelerator.log(log, step=step)
                        training_log.append(log)
                        with open(
                            os.path.join(args.output_dir, "training.log"), "w"
                        ) as f:
                            ujson.dump(training_log, f, indent=2)

                    if args.save_steps > 0 and step % args.save_steps == 0:
                        ckpt_path = save_checkpoint(
                            args, accelerator, colt5, optimizer, scheduler, step
                        )
                        if ckpt_path is not None:
                            checkpoint_path_list.append(ckpt_path)

        accelerator.wait_for_everyone()
        accelerator.print(f"#> Epoch {epoch} ended.")

    final_ckpt_path = save_checkpoint(args, accelerator, colt5)
    checkpoint_path_list.append(final_ckpt_path)

    return checkpoint_path_list


def index(args, model_ckpt_path=None, collection=None):
    args.checkpoint = model_ckpt_path or args.checkpoint
    ckpt_id = get_checkpoint_id(args.checkpoint)
    print_message("#> indexing using model with ckpt_id:", ckpt_id)
    checkpoint_dir = os.path.dirname(args.checkpoint)

    index_name = f"index_{ckpt_id}.faiss"

    if exist_file_with_prefix(args.output_dir, index_name) or exist_file_with_prefix(
        checkpoint_dir, index_name
    ):
        print_message("#> index already exists, will not create a new one")
        return

    index_path = os.path.join(args.output_dir, index_name)

    collection_ckpt_name = f"collection_{ckpt_id}.hdf5"
    collection_ckpt_path = os.path.join(args.output_dir, collection_ckpt_name)

    if os.path.exists(collection_ckpt_path):
        print_message("#> Find collection checkpoint, loading...")
        f = h5py.File(collection_ckpt_path, "r")

        assert f.attrs["id"] == ckpt_id, f.attrs["id"]

        collection_vectors = f["collection/vectors"][:]
        print_message("#> Loaded collection from:", collection_ckpt_path)
    else:
        colt5 = ColT5(
            model_name_or_path=args.model_name_or_path,
            doc_maxlen=args.doc_maxlen,
            # mask_punctuation=args.mask_punctuation,
        )
        load_checkpoint(args.checkpoint, model=colt5)
        colt5.to(DEVICE)
        colt5.eval()

        with Timer(description="Loading collections..."):
            collections = DocDataset(args, collection=collection)
        collection_loader = DataLoader(
            collections,
            batch_size=args.index_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collections.collate_fn,
            drop_last=False,
        )

        print_message("#> Loaded collections from:", args.collection)

        collection_vectors = []
        collection_doclens = {}
        collection_idx2pid = []
        for idx, batch in tqdm(
            enumerate(collection_loader), total=len(collection_loader)
        ):
            with torch.no_grad():
                pids, passages = batch
                D = colt5.doc(*passages, keep_dims=False)
                assert type(D) == list
                assert len(D) == len(pids)

                idx2pid = sum([[pid] * d.size(0) for pid, d in zip(pids, D)], [])
                doclens = {pid: d.size(0) for pid, d in zip(pids, D)}
                embs = torch.cat(D).float().cpu().tolist()

                assert len(idx2pid) == len(embs)

                collection_vectors.extend(embs)
                collection_doclens.update(doclens)
                collection_idx2pid.extend(idx2pid)

        collection_vectors = np.array(collection_vectors, dtype=np.float32)

        f = h5py.File(collection_ckpt_path, "w")
        f.attrs["id"] = ckpt_id
        f.create_dataset("collection/vectors", data=collection_vectors)
        f.create_dataset("collection/idx2pid", data=collection_idx2pid)

        del colt5
        torch.cuda.empty_cache()

    print_message("#> collection_vectorrs.shape:", collection_vectors.shape)

    sample_fraction = args.sample_fraction
    training_sample = collection_vectors.copy()
    if sample_fraction:
        sample_size = int(sample_fraction * len(collection_vectors))
        sample_idx = np.random.choice(
            np.arange(len(collection_vectors)), size=sample_size, replace=False
        )
        training_sample = training_sample[sample_idx]

    print_message("#> training_sample.shape:", training_sample.shape)

    dim = training_sample.shape[1]
    partitions = 1 << math.ceil(math.log2(8 * math.sqrt(len(collection_vectors))))
    print_message("#> partitions (nlist) of faiss index:", partitions)
    index = FaissIndex(dim, partitions=partitions)

    with Timer(description="Index training"):
        index.train(training_sample)

    with Timer("#> Adding collection vectors to index..."):
        index.add(collection_vectors)
    # for start_idx in range(0, len(collection_vectors), args.batch_size):
    #     index.add(collection_vectors[start_idx: start_idx+args.batch_size)
    index.save(index_path)


def retrieve(
    args,
    model_ckpt_path=None,
    qrels=None,
    queries=None,
    collection=None,
    candidates=None,
):
    args.checkpoint = model_ckpt_path or args.checkpoint
    ckpt_id = get_checkpoint_id(args.checkpoint)
    print_message("#> retrieve using model with ckpt_id:", ckpt_id)
    checkpoint_dir = os.path.dirname(args.checkpoint)

    index_name = f"index_{ckpt_id}.faiss"

    # get index directory
    index_dir = args.output_dir
    if exist_file_with_prefix(args.output_dir, index_name):
        index_dir = args.output_dir
    elif exist_file_with_prefix(checkpoint_dir, index_name):
        index_dir = checkpoint_dir
    else:
        print_message("#> index not found, will create one")
        index(args, model_ckpt_path=model_ckpt_path, collection=collection)
        index_dir = args.output_dir

    index_path = os.path.join(index_dir, index_name)

    collection_ckpt_name = f"collection_{ckpt_id}.hdf5"
    collection_ckpt_path = os.path.join(index_dir, collection_ckpt_name)

    if not os.path.exists(collection_ckpt_path):
        print_message("#> collection ckpt not found, will create one")
        index(args, model_ckpt_path=model_ckpt_path)

    f = h5py.File(collection_ckpt_path, "r")
    vectors = f["collection/vectors"][:]
    idx2pid = f["collection/idx2pid"][:]
    print_message("#> Loaded collection from ", collection_ckpt_path)

    print_message("#> Sanity check...")

    assert len(idx2pid) == len(vectors)

    with Timer(description="unique_consecutive idx2doc"):
        all_pids, idx2doc, doclens = torch.unique_consecutive(
            torch.tensor(idx2pid), return_counts=True, return_inverse=True
        )
    print_message("#> num of passages", len(all_pids))

    idx2doclens = doclens[idx2doc]

    offset = [0] + list(accumulate(doclens.tolist()))
    pid2idx = {all_pids[i].item(): offset[i] for i in range(len(all_pids))}

    doc_vec = torch.zeros(
        len(vectors) + args.doc_maxlen, len(vectors[0]), dtype=torch.float32
    )
    doc_vec[: len(vectors)] = torch.tensor(vectors, dtype=torch.float32)
    print_message("#> Loaded doc_token_vec:", doc_vec.size())

    with Timer(description="strided_doc_vec"):
        stride = doclens.max().item()
        outdim = doc_vec.size(0) - stride + 1
        dim = doc_vec.size(1)
        strided_doc_vec = torch.as_strided(
            doc_vec, (outdim, stride, dim), (dim, dim, 1)
        )

    print_message("#> Loading model...")
    colt5 = ColT5(
        model_name_or_path=args.model_name_or_path,
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=args.dim,
        similarity_metric=args.similarity_metric,
    )

    with Timer(description=f"Load checkpoint from {args.checkpoint}"):
        load_checkpoint(args.checkpoint, model=colt5)

    colt5.to(DEVICE)
    colt5.eval()

    with Timer(description="load queries"):
        qrels = QrelDataset(args, qrels=qrels, queries=queries)
    qrel_loader = DataLoader(
        qrels,
        batch_size=args.retrieval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=qrels.collate_fn,
        drop_last=False,
    )

    with Timer(description=f"Load faiss index from {index_path}"):
        faiss_index = FaissRetrieveIndex(
            faiss_index_path=index_path,
            emb2pid=idx2pid,
            nprobe=args.nprobe,
        )

    print_message("#> Loaded queries:", len(qrels))

    output_pids = []
    target_pids = []
    output_qids = []
    output_scores = []

    print_message(f"#> {len(qrel_loader)} batches")

    for idx, batch in tqdm(enumerate(qrel_loader), total=len(qrel_loader)):
        qids, pids, queries = batch
        with torch.no_grad():
            with Timer(description="query encoding"):
                Q = colt5.query(*queries)
        n_queries, n_tokens, dim = Q.size()
        assert (
            n_queries == len(qids) and n_tokens == args.query_maxlen and dim == args.dim
        )

        if candidates is not None:
            retrieved_pids = [candidates[qid.item()] for qid in qids]
        else:
            with Timer(description=f"Faiss retrieve batch idx {idx}"):
                retrieved_pids = faiss_index.retrieve(args.faiss_depth, Q, verbose=True)

        with Timer(description="flatten"):
            unordered_candidates = []
            for qoffset, ranking in zip(range(len(qids)), retrieved_pids):
                unordered_candidates.extend(
                    [(qoffset, pid2idx[pid]) for pid in ranking]
                )

            qoffsets, poffsets = zipstar(unordered_candidates)

        with Timer(description="scoring"):
            qoffsets = torch.tensor(qoffsets, device=DEVICE)
            poffsets = torch.tensor(poffsets)

            D = strided_doc_vec[poffsets]
            assert D.size() == (len(poffsets), stride, dim)
            D = D.to(DEVICE)

            Q = Q[qoffsets]
            assert Q.size() == (len(qoffsets), n_tokens, dim)
            Q = Q.permute(0, 2, 1)

            mask = torch.arange(stride, device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= idx2doclens[poffsets].to(DEVICE).unsqueeze(-1)

            scores = (D @ Q) * mask.unsqueeze(-1)

            scores = scores.max(1).values.sum(-1)

        output_pids.extend(all_pids[idx2doc[poffsets.cpu()]].tolist())
        output_qids.extend(qids[qoffsets.cpu()].tolist())
        target_pids.extend(pids[qoffsets.cpu()].tolist())
        output_scores.extend(scores.cpu().tolist())
        torch.cuda.empty_cache()

    print_message("#> Finished ranking")

    last_group = None
    ranking_result = []
    for qid, target_pid, pid, score in zip(
        output_qids, target_pids, output_pids, output_scores
    ):
        if qid != last_group:
            ranking_result.append(
                {
                    "qid": qid,
                    "target_pid": target_pid,
                    "ranking_result": [],
                }
            )
        ranking_result[-1]["ranking_result"].append((pid, score))
        last_group = qid

    result_file_name = f"result_{ckpt_id}.json"
    result_file_path = os.path.join(args.output_dir, result_file_name)
    with open(result_file_path, "w") as f:
        ujson.dump(ranking_result, f, indent=2)

    return ranking_result


def parse_args():
    parser = ArgumentParser("ColT5 / ColBERT training")

    parser.add_argument("--project", type=str, default="ColT5-Training")
    parser.add_argument(
        "--pipeline", type=str, default="train", choices=["train", "index", "retrieve"]
    )

    parser.add_argument("--model_name_or_path", type=str, default="t5-base")

    parser.add_argument("--triples", type=str, default=None)
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--qrels", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--candidates", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="corpus")

    parser.add_argument("--query_maxlen", type=int, default=32)
    parser.add_argument("--doc_maxlen", type=int, default=128)
    parser.add_argument("--rm_stopwords", action="store_true")
    parser.add_argument("--rm_punctuation", action="store_true")
    parser.add_argument("--lower", action="store_true")
    parser.add_argument(
        "--similarity_metric", type=str, choices=["cosine", "l2"], default="cosine"
    )
    parser.add_argument("--dim", type=int, default=128)

    parser.add_argument("--in_batch_negative", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--index_batch_size", type=int, default=32)
    parser.add_argument("--retrieval_batch_size", type=int, default=32)
    parser.add_argument("--accumsteps", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_epochs", type=int, default=0)
    parser.add_argument("--tot_n_ckpts", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--with_track", action="store_true")
    parser.add_argument("--use_training_progress_bar", action="store_true")

    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--sample_fraction", type=float, default=0.2)

    parser.add_argument("--nprobe", type=int, default=10)
    parser.add_argument("--faiss_depth", type=int, default=100)

    parser.add_argument("--at", nargs="*", type=int, default=[1, 5, 10, 50])

    args = parser.parse_args()

    assert os.path.exists(args.model_name_or_path)
    assert os.path.exists(args.triples)
    assert os.path.exists(args.queries)
    assert os.path.exists(args.collection)

    if args.tot_n_ckpts > 1 and args.save_steps > 0:
        print_message("#> Warning: tot_n_ckpts > 1, save_steps will be ignored")
    if args.in_batch_negative and args.similarity_metric == "l2":
        raise NotImplementedError
    if args.pipeline == "train":
        assert args.triples is not None
        assert args.queries is not None
        assert args.collection is not None
    if args.pipeline != "train":
        assert args.checkpoint is not None
    if args.pipeline == "index":
        assert args.collection is not None
    if args.pipeline == "retrieve":
        assert args.qrels is not None
        assert args.queries is not None

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
    triples = load_triples(args.triples)
    qrels = load_qrel(args.qrels)
    candidates = load_candidates(args.candidates)

    collection = clean(
        collection,
        rm_stopwords=args.rm_stopwords,
        rm_punctuation=args.rm_punctuation,
        lower=args.lower,
        data_name=args.data_name,
    )

    if args.pipeline == "train":
        checkpoint_path_list = train(
            args, collection=collection, triples=triples, queries=queries
        )
        # currently only test the last checkpoint
        final_ckpt_path = checkpoint_path_list[-1]
        index(args, model_ckpt_path=final_ckpt_path, collection=collection)
        ranking_result = retrieve(
            args,
            model_ckpt_path=final_ckpt_path,
            qrels=qrels,
            queries=queries,
            collection=collection,
            candidates=candidates,
        )
        eval_result(
            results=ranking_result,
            output_path=args.output_dir,
            model_ckpt_path=final_ckpt_path,
            at=args.at,
        )
    elif args.pipeline == "index":
        index(args, collection=collection)
        ranking_result = retrieve(
            args,
            qrels=qrels,
            queries=queries,
            collection=collection,
            candidates=candidates,
        )
        eval_result(
            results=ranking_result,
            output_path=args.output_dir,
            model_ckpt_path=args.checkpoint,
            at=args.at,
        )
    elif args.pipeline == "retrieve":
        ranking_result = retrieve(
            args,
            qrels=qrels,
            queries=queries,
            collection=collection,
            candidates=candidates,
        )
        eval_result(
            results=ranking_result,
            output_path=args.output_dir,
            model_ckpt_path=args.checkpoint,
            at=args.at,
        )
