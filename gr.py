# gr.py
# plain gr with title or prefix as identifier, same as mvdr
# without contrastive loss (wo negative sampling)

from collections import defaultdict
import os
import time
import ujson
import torch
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
    load_candidates,
    zipstar,
    file_tqdm,
    load_qrel,
    load_triples,
    load_queries,
    print_message,
    save_checkpoint,
    load_checkpoint,
    load_collection,
    create_directory,
    get_checkpoint_id,
    exist_file_with_prefix,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MarisaTrie(object):
    def __init__(
        self,
        sequences=[],  # list of list of int
        cache_fist_branch=True,
        max_token_id=256001,
    ):
        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        )
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

        self.cache_fist_branch = cache_fist_branch
        if self.cache_fist_branch:
            self.zero_iter = list({sequence[0] for sequence in sequences})
            assert len(self.zero_iter) == 1
            self.first_iter = list({sequence[1] for sequence in sequences})

        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence]) for sequence in sequences
        )

    def get(self, prefix_sequence):  # list of int
        if self.cache_fist_branch and len(prefix_sequence) == 0:
            return self.zero_iter
        elif (
            self.cache_fist_branch
            and len(prefix_sequence) == 1
            and self.zero_iter == prefix_sequence
        ):
            return self.first_iter
        else:
            key = "".join([self.int2char[i] for i in prefix_sequence])
            return list(
                {
                    self.char2int[e[len(key)]]
                    for e in self.trie.keys(key)
                    if len(e) > len(key)
                }
            )

    def __iter__(self):
        for sequence in self.trie.iterkeys():
            yield [self.char2int[e] for e in sequence]

    def __len__(self):
        return len(self.trie)

    def __getitem__(self, value):
        return self.get(value)


def _do_span_wrapper(args):
    return args[0], do_span(**args[1])


class TrainDataset(Dataset):
    def __init__(self, args, triples=None, queries=None, collection=None):
        self.span = args.use_span
        self.n_span = args.n_span
        self.ngram = args.ngram
        self.data_name = args.data_name
        self.num_workers = args.num_workers
        self.temperature = args.temperature
        self.span_length = args.span_length

        self.tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)
        self.query_maxlen = args.query_maxlen
        self.doc_maxlen = args.doc_maxlen

        self.eos_token_id = self.tok.eos_token_id

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
        if self.span:  # split collection corpus into spans
            self._do_span()

    def _generate_cache_file_name(self, cache_dir="cache"):
        cache_file_name = f"span_{self.data_name}_cache_{self.n_span}_{self.ngram}_{self.temperature}_{self.span_length}.list"
        cache_file_path = os.path.join(cache_dir, cache_file_name)
        create_directory(cache_dir)
        return cache_file_path

    def _do_span(self):
        cache_file = self._generate_cache_file_name()

        if os.path.exists(cache_file):
            print_message("#> Loading cached spans...")
            with open(cache_file, "rb") as f:
                self.triples_span = pickle.load(f)
            return
        else:
            if len(self.triples) < 100000:
                print_message("#> Generating spans...")
                self.triples_span = []

                for qid, pid, *_ in tqdm(self.triples):
                    query, passage = self.queries[qid], self.collection[pid]
                    spans = do_span(
                        query,
                        passage,
                        n_samples=self.n_span,
                        ngram=self.ngram,
                        temperature=self.temperature,
                        min_length=self.span_length,
                        max_length=self.span_length,
                    )
                    for span in spans:
                        self.triples_span.append((qid, span))
            else:
                print_message("#> Generating spans using multiprocessing...")
                extracted_args = [
                    (
                        qid,
                        {
                            "query": self.queries[qid],
                            "passage": self.collection[pid],
                            "n_samples": self.n_span,
                            "ngram": self.ngram,
                            "temperature": self.temperature,
                            "min_length": self.span_length,
                            "max_length": self.span_length,
                        },
                    )
                    for qid, pid, *_ in self.triples
                ]
                print_message("#> Start multiprocessing...")
                self.triples_span = []
                with Pool(self.num_workers) as pool:
                    for qid, spans in pool.imap(_do_span_wrapper, extracted_args):
                        for span in spans:
                            self.triples_span.append((qid, span))

            with open(cache_file, "wb") as f:
                pickle.dump(self.triples_span, f)

    def __len__(self):
        return len(self.triples) if not self.span else len(self.triples_span)

    def __getitem__(self, index):
        if self.span:
            qid, span = self.triples_span[index]
            query = self.queries[qid]

            return query, span
        else:
            query, pos = self.triples[index]
            query, pos = (
                self.queries[query],
                self.collection[pos],
            )

            return query, pos

    def collate_fn(self, batch):
        query, pos = zip(*batch)
        query, pos = list(query), list(pos)
        # n_samples = len(query)

        # tensorize query
        query_obj = self.tok(
            query,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.query_maxlen,
        )
        query_ids, query_mask = query_obj["input_ids"], query_obj["attention_mask"]
        # query_ids[query_ids == 0] = self.eos_token_id

        # tensorize pos and neg
        doc_obj = self.tok(
            pos,
            padding="longest",
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.doc_maxlen,
        )
        doc_ids, doc_mask = doc_obj["input_ids"], doc_obj["attention_mask"]
        doc_ids[doc_ids == 0] = -100

        # use pos only
        Q = (query_ids, query_mask)
        D = (doc_ids, doc_mask)

        return Q, D


class QrelDataset(Dataset):
    def __init__(
        self, args, qrels=None, queries=None, collection=None, candidates=None
    ):
        super().__init__()

        self.tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)

        self.query_maxlen = args.query_maxlen
        self.doc_maxlen = args.doc_maxlen

        self.eos_token_id = self.tok.eos_token_id

        self.queries = queries or load_queries(args.queries)
        self.qrel = qrels or load_qrel(args.qrels)

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

        self.has_candidate = False
        if candidates is not None:
            self.candidates = candidates
            self.has_candidate = True
            self.qrel = self._expand_qrel()

        if args.debug:
            self.qrel = self.qrel[:100]

    def _expand_qrel(self):
        assert self.has_candidate
        expanded_qrel = []

        for qid, pid in self.qrel:
            candidates = self.candidates[qid]
            for can in candidates:
                spans = do_span(
                    self.queries[qid],
                    self.collection[can],
                    n_samples=3,
                    ngram=3,
                    temperature=1.0,
                    min_length=10,
                    max_length=10,
                )
                for span in spans:
                    expanded_qrel.append((qid, span, can, pid))
            # expanded_qrel.extend([(qid, can, pid) for can in candidates])

        return expanded_qrel

    def __len__(self):
        return len(self.qrel)

    def __getitem__(self, index):
        qrel = self.qrel[index]
        if self.has_candidate:
            qid, span, can_pid, pid = qrel
            query = self.queries[qid]
            return qid, can_pid, query, span, pid

        qid, pid = qrel
        query = self.queries[qid]
        passage = self.collection[pid]

        return qid, pid, query, passage

    def collate_fn(self, data):
        qids, pids, queries, passages, *_ = zipstar(data)

        self.tok: T5TokenizerFast

        queries_tok = self.tok(
            queries,
            max_length=self.query_maxlen,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        assert queries_tok["input_ids"].size(0) == len(qids)

        # queries_tok["input_ids"][queries_tok["input_ids"] == 0] = self.eos_token_id

        if self.has_candidate:
            target_pids = _[0]
            passage_tok = self.tok(
                passages,
                max_length=self.doc_maxlen,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            passage_tok["input_ids"][passage_tok["input_ids"] == 0] = -100

            return (
                qids,
                queries,
                (queries_tok["input_ids"], queries_tok["attention_mask"]),
                pids,
                passages,
                passage_tok["input_ids"],
                target_pids,
            )

        return (
            qids,
            queries,
            (queries_tok["input_ids"], queries_tok["attention_mask"]),
            pids,
            passages,
        )


def build_sealsearcher(fm_index_path, model, tokenizer, args):
    fm_index = SEALSearcher.load_fm_index(fm_index_path)
    model.config.forced_bos_token_id = None

    # for some trained models, the mask logit is set to 0 for some reason. This ugly hack fixes it, see seal/retrieval.py:L583
    if hasattr(model, "final_logits_bias"):
        model.config.add_bias_logits = True
        model.final_logits_bias[0, tokenizer.pad_token_id] = float("-inf")
        model.final_logits_bias[0, tokenizer.bos_token_id] = float("-inf")
        model.final_logits_bias[0, tokenizer.mask_token_id] = float("-inf")

    searcher = SEALSearcher(
        fm_index=fm_index,
        bart_tokenizer=tokenizer,
        bart_model=model,
        backbone="t5",
        length=args.span_length,
        use_markers=False,
        value_conditioning=False,
        decode_titles=False,
        jobs=args.num_workers * 2,
        beam=5,
    )

    return searcher


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

    t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    with Timer(description="Loading dataset"):
        dataset = TrainDataset(
            args, triples=triples, queries=queries, collection=collection
        )

    accelerator.print(f"#> Loaded {len(dataset)} triples.")
    dataloader = torch.utils.data.DataLoader(
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

    optimizer = AdamW(filter(lambda p: p.requires_grad, t5.parameters()), lr=args.lr)
    scheduler = get_constant_schedule(optimizer)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=500,
    #     num_training_steps=(len(dataloader) * args.n_epochs) // args.accumsteps,
    # )

    t5, optimizer, dataloader, scheduler = accelerator.prepare(
        t5, optimizer, dataloader, scheduler
    )
    # t5 = torch.compile(t5)
    t5.train()

    if args.with_track:
        accelerator.init_trackers(args.project, args.__dict__)

    start_time = time.time()
    step = 0
    training_log = []
    checkpoint_path_list = []

    for epoch in range(args.n_epochs):
        accelerator.print(f"#> Epoch {epoch} started.")
        accelerator.wait_for_everyone()

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

        optimizer.zero_grad(set_to_none=True)

        for idx, batch in enumerate(iterator):
            step += 1
            with accelerator.accumulate(t5):
                query, passage = batch
                input_dict = {
                    "input_ids": query[0],
                    "attention_mask": query[1],
                    "labels": passage[0],
                }
                output = t5(**input_dict)

                loss = output.loss

                accelerator.backward(loss)
                if args.max_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        parameters=t5.parameters(),
                        max_norm=args.max_grad_norm,
                        norm_type=2,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    loss = accelerator.gather(loss).mean().item()
                    if step % 10 == 0 and accelerator.is_local_main_process:
                        loss_report.append(loss)
                        with torch.no_grad():
                            ma_loss = np.mean(loss_report[-100:], axis=0)
                            cur_loss = loss_report[-1]

                        if args.use_training_progress_bar:
                            iterator.set_postfix_str(
                                f"Moving Average Loss {ma_loss} | CurLoss {cur_loss}"
                            )

                        duration = time.time() - start_time
                        accelerator.print(
                            f"#> Epoch {epoch} | Step {idx} || CurLoss {cur_loss} | "
                            f"Average Loss {ma_loss}"
                            f"Time {duration}s | Rest {(duration / (step + 1)) * (tot_steps - step - 1) / 60:.2f}min"
                        )
                        log = {
                            "epoch": epoch,
                            "step": step,
                            "cur_loss": cur_loss,
                            "ma_loss": ma_loss,
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
                            args, accelerator, t5, optimizer, scheduler, step
                        )
                        if ckpt_path is not None:
                            checkpoint_path_list.append(ckpt_path)

        accelerator.wait_for_everyone()
        accelerator.print(f"#> Epoch {epoch} ended.")

    final_ckpt_path = save_checkpoint(args, accelerator, t5)
    checkpoint_path_list.append(final_ckpt_path)

    return checkpoint_path_list, accelerator


def index(args, model_ckpt_path=None, collection=None):
    # trie = MarisaTrie([[0]+tokenizer.encode(‘Hello World’)])

    # output = model.generate(tokenizer.encode(‘Hello World’, return_tensors=‘pt’), prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()))

    args.checkpoint = model_ckpt_path or args.checkpoint
    ckpt_id = get_checkpoint_id(args.checkpoint)
    print_message("#> indexing using model with ckpt_id:", ckpt_id)

    index_name = f"index_{ckpt_id}"
    index_path = os.path.join(args.output_dir, index_name)

    checkpoint_dir = os.path.dirname(args.checkpoint)
    if exist_file_with_prefix(args.output_dir, index_name) or exist_file_with_prefix(
        checkpoint_dir, index_name
    ):
        # if os.path.exists(index_path):
        print_message("#> index already exists, will not create a new one")
        return

    tokenizer = T5TokenizerFast.from_pretrained(args.model_name_or_path)

    if collection is None:
        print_message("#> Loading collection...")
        collection = {}

        with open(args.collection) as f:
            for line in file_tqdm(f):
                pid, passage, *_ = line.strip().split("\t")

                assert len(passage) >= 1
                if len(_) >= 1:
                    title, *_ = _
                    passage = title + " | " + passage

                pid = int(pid)
                collection[pid] = passage

        collection = clean(
            collection,
            rm_punctuation=args.rm_punctuation,
            rm_stopwords=args.rm_stopwords,
            lower=args.lower,
            data_name=args.data_name,
        )

        print_message(f"#> load {len(collection)} passages")

    collection_token_ids = []
    labels = []

    for pid, passage in tqdm(
        collection.items(), total=len(collection), desc="tokenize"
    ):
        passage_token_ids = [0] + tokenizer.encode(
            passage, add_special_tokens=False
        )  # decoder start token id
        passage_token_ids += [tokenizer.eos_token_id]
        collection_token_ids.append(passage_token_ids)
        labels.append(pid)

    if args.use_span:
        # using FM index
        index = FMIndex()
        index.initialize(collection_token_ids, in_memory=True)
        index.labels = labels

        index.save(index_path)
        print_message(f"#> Saving FM index to {index_path} ...")
    else:
        # using prefix constraint
        with Timer("build trie"):
            trie = MarisaTrie(collection_token_ids)

        print_message(f"#> Saving trie to {index_path} ...")
        pickle.dump(trie, open(index_path, "wb"))


def rank(
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

    index_name = f"index_{ckpt_id}"
    index_dir = args.output_dir
    # index_path = os.path.join(index_dir, index_name)

    if not exist_file_with_prefix(index_dir, index_name):
        print_message(
            "#> index not found in {}, will search in checkpoint dir".format(index_dir)
        )
        index_dir = os.path.dirname(args.checkpoint)

    if not exist_file_with_prefix(index_dir, index_name):
        print_message(
            "#> index not found in {}, will create a new one in {}".format(
                index_dir, args.output_dir
            )
        )
        index(args, model_ckpt_path=model_ckpt_path, collection=collection)
        index_dir = args.output_dir

    index_path = os.path.join(index_dir, index_name)

    t5 = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        repetition_penalty=None,
        no_repeat_ngram_size=0,
        encoder_no_repeat_ngram_size=0,
        bad_words_ids=None,
        min_length=args.span_length,
        max_length=args.span_length,
        eos_token_id=None,
        forced_eos_token_id=None,
        num_beams=15,
        num_beam_groups=1,
        diversity_penalty=0.0,
        remove_invalid_values=True,
    )

    tok = T5TokenizerFast.from_pretrained(args.model_name_or_path)
    with Timer(description="Loading model"):
        load_checkpoint(args.checkpoint, model=t5)

    t5.to(DEVICE)
    t5.eval()
    # t5 = torch.compile(t5)

    if args.use_span:
        seal_searcher = build_sealsearcher(index_path, t5, tok, args)
    else:
        with Timer("Loading trie"):
            trie = pickle.load(open(index_path, "rb"))

    with Timer(description="Loading dataset"):
        dataset = QrelDataset(
            args,
            qrels=qrels,
            queries=queries,
            collection=collection,
            candidates=candidates,
        )
    print_message(f"#> Loaded {len(dataset)} triples.")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.retrieval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    rank_list = []

    if candidates is not None:
        ce = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        rank_dict = defaultdict(list)
        target_dict = {}

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        qids, queries, queries_pt, pids, passages, *_ = batch
        # qids, pids, queries, passages = batch
        if candidates is not None:
            queries_pt = (queries_pt[0].to(DEVICE), queries_pt[1].to(DEVICE))
            passage_pt, target_pids = _
            passage_pt = passage_pt.to(DEVICE)

            with torch.no_grad():
                lm_logits = t5(
                    input_ids=queries_pt[0],
                    attention_mask=queries_pt[1],
                    labels=passage_pt,
                    return_dict=True,
                ).logits

                # indices = passage_pt.unsqueeze(2)
                # indices[indices == -100] = 0

                # gathered_logits = torch.gather(lm_logits, 2, indices).squeeze(2)
                # assert gathered_logits.size() == (passage_pt.size(0), passage_pt.size(1))
                # gathered_logits[passage_pt == -100] = 0

                # averaged_score = gathered_logits.sum(dim=-1) / (passage_pt != -100).sum(dim=-1)
                # assert averaged_score.size() == (passage_pt.size(0),)
                # averaged_score = averaged_score.tolist()
                ce_loss = ce(
                    lm_logits.view(-1, lm_logits.size(-1)), passage_pt.view(-1)
                )
                # assert logits.size() == (args.batch_size, args.doc_maxlen, 50264)

                ce_loss = ce_loss.view(passage_pt.size(0), passage_pt.size(1))
                # calculate the average ignoring the loss where passage_pt == -100
                # set the loss to zero where passage_pt == -100
                ce_loss[passage_pt == -100] = 0
                ce_loss = -ce_loss.sum(dim=-1) / (passage_pt != -100).sum(dim=-1)
                assert ce_loss.size() == (passage_pt.size(0),)
                averaged_score = ce_loss.tolist()

                for qid, pid, score, target_pid in zip(
                    qids, pids, averaged_score, target_pids
                ):
                    rank_dict[qid].append((pid, score))
                    target_dict[qid] = target_pid

            continue

        if args.use_span:
            search_result = seal_searcher.batch_search(queries, k=args.topk)
            for qid, target_pid, docs in zip(qids, pids, search_result):
                rank_result = {
                    "qid": qid,
                    "target_pid": target_pid,
                    "ranking_result": [(doc.docid, doc.score) for doc in docs],
                }
                rank_list.append(rank_result)
        else:
            queries_pt = (queries_pt[0].to(DEVICE), queries_pt[1].to(DEVICE))

            with torch.no_grad():
                output = t5.generate(
                    input_ids=queries_pt[0],
                    attention_mask=queries_pt[1],
                    max_length=args.doc_maxlen,
                    num_beams=args.topk,
                    num_return_sequences=args.topk,
                    length_penalty=0,
                    early_stopping=False,
                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(
                        sent.tolist()
                    ),
                )
                output = tok.batch_decode(output, skip_special_tokens=True)
                output = [
                    [str(x).strip() for x in output[k : k + args.topk]]
                    for k in range(0, len(output), args.topk)
                ]
                assert len(output) == len(qids) and len(output[0]) == args.topk
                # labels = tok.batch_decode(passages[0], skip_special_tokens=True)
                labels = [str(x).strip() for x in passages]

                rank_sub_list = [
                    {
                        "qid": qid,
                        "target_pid": pid,
                        "target_passage": label,
                        "ranking_result": passage_list,
                    }
                    for qid, pid, label, passage_list in zip(qids, pids, labels, output)
                ]
                rank_list.extend(rank_sub_list)

    if candidates is not None:
        rank_list = []
        for qid, pid_score_list in rank_dict.items():
            score_reduce = defaultdict(float)
            for pid, score in pid_score_list:
                score_reduce[pid] += score
            pid_score_list = [(pid, score) for pid, score in score_reduce.items()]
            pid_score_list = sorted(pid_score_list, key=lambda x: x[1], reverse=True)
            rank_result = {
                "qid": qid,
                "target_pid": target_dict[qid],
                "ranking_result": pid_score_list,
            }
            rank_list.append(rank_result)

    result_file_name = f"result_{ckpt_id}.json"
    with open(os.path.join(args.output_dir, result_file_name), "w") as f:
        ujson.dump(rank_list, f, indent=2)

    return rank_list


def parse_args():
    parser = ArgumentParser("t5 training")

    parser.add_argument("--project", type=str, default="t5-Training")
    parser.add_argument(
        "--pipeline", type=str, default="train", choices=["train", "index", "retrieve"]
    )

    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--triples", type=str, required=True)
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

    parser.add_argument("--use_span", action="store_true")
    parser.add_argument("--n_span", type=int, default=10)
    parser.add_argument("--span_length", type=int, default=10)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--retrieval_batch_size", type=int, default=32)
    parser.add_argument("--accumsteps", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--tot_n_ckpts", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--with_track", action="store_true")
    parser.add_argument("--use_training_progress_bar", action="store_true")

    parser.add_argument("--at", nargs="*", type=int, default=[1, 5, 10, 50])
    parser.add_argument("--topk", type=int, default=100)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # args check
    assert os.path.exists(args.model_name_or_path)
    assert os.path.exists(args.triples)
    assert os.path.exists(args.queries)
    assert os.path.exists(args.collection)

    assert "t5" in args.model_name_or_path, "only support t5 model"

    if args.tot_n_ckpts > 1 and args.save_steps > 0:
        print_message("#> Warning: tot_n_ckpts > 1, save_steps will be ignored")
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
        assert args.collection is not None

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
        checkpoint_path_list, accelerator = train(
            args, collection=collection, triples=triples, queries=queries
        )
        # currently only test the last checkpoint
        if accelerator.is_main_process:
            final_ckpt_path = checkpoint_path_list[-1]
            index(args, model_ckpt_path=final_ckpt_path, collection=collection)
            ranking_result = rank(
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
        ranking_result = rank(
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
        ranking_result = rank(
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
