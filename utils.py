# utils.py
import os
import re
import math
import tqdm
import ftfy
import time
import json
import torch
import random
import string
import datetime
import itertools
import shortuuid
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from collections import OrderedDict, defaultdict


def print_message(*s, condition=True):
    s = " ".join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        print(msg, flush=True)

    return msg


def timestamp():
    format_str = "%Y-%m-%d_%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


def file_tqdm(file):
    print(f"#> Reading {file.name}")

    with tqdm.tqdm(
        total=os.path.getsize(file.name) / 1024.0 / 1024.0, unit="MiB"
    ) as pbar:
        for line in file:
            yield line
            pbar.update(len(line) / 1024.0 / 1024.0)

        pbar.close()


def get_checkpoint_id(path):
    assert os.path.exists(path), f"Checkpoint path does not exist: {path}"
    return path.split("_")[-1].split(".")[0]


def save_checkpoint(
    args, accelerator, model, optimizer=None, scheduler=None, step=None
):
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        ckpt_id = shortuuid.uuid()
        ckpt_file = (
            f"checkpoint_final-{ckpt_id}.pt"
            if step is None
            else f"checkpoint_{step}-{ckpt_id}.pt"
        )
        checkpoint_path = os.path.join(args.output_dir, ckpt_file)
        os.makedirs(args.output_dir, exist_ok=True)

        unwrap_model = accelerator.unwrap_model(model)
        checkpoint = {}
        checkpoint["model"] = unwrap_model.state_dict()
        checkpoint["id"] = ckpt_id
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        if step is not None:
            checkpoint["step"] = step
        checkpoint["args"] = args.__dict__

        accelerator.save(checkpoint, checkpoint_path)
        accelerator.print("#> Saved checkpoint:", checkpoint_path)
        return checkpoint_path


def load_checkpoint(path, model, optimizer=None, return_step=False, verbose=True):
    ckpt_id = path.split("-")[-1].split(".")[0]
    if verbose:
        print_message("#> Loading checkpoint", path, "..")

    checkpoint = torch.load(path, map_location="cpu")
    assert ckpt_id == checkpoint["id"], "Checkpoint ID does not match!"

    state_dict = checkpoint["model"]

    # if verbose:
    #     print_message("#> Loaded module keys:", state_dict.keys())

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k.startswith("module."):
            name = k[7:]
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict)
    except:
        print_message(
            "[WARNING] Failed to load state dict. Trying to load without strict."
        )
        model.load_state_dict(new_state_dict, strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if verbose:
        print_message("#> Loaded checkpoint from:", path)
        print_message("#> model information:\n", checkpoint["args"])

    if return_step and "step" in checkpoint:
        return checkpoint["step"]


def create_directory(path):
    if os.path.exists(path):
        print("\n")
        print_message("#> Note: Output directory", path, "already exists\n\n")
    else:
        print("\n")
        print_message("#> Creating directory", path, "\n\n")
        os.makedirs(path)


def load_collection(path, return_dict=True):
    """
    returns: list of {pid, passage} or dict of {pid: passage}
    default: return_dict=True
    """
    print_message("#> Loading collection...")

    if return_dict:
        collection = {}
    else:
        collection = []

    with open(path) as f:
        for line in file_tqdm(f):
            pid, passage, *_ = line.strip().split("\t")

            assert len(passage) >= 1
            if len(_) >= 1:
                title, *_ = _
                passage = title + " | " + passage

            pid = int(pid)
            if return_dict:
                collection[pid] = passage
            else:
                collection.append({"pid": pid, "passage": passage})

    return collection


def load_queries(path):
    """
    returns: dict of {qid: query}
    """
    print_message("#> Loading queries...")

    queries = {}

    with open(path) as f:
        for line in f:
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query

    return queries


def load_triples(path):
    """
    returns: list of (qid, pos, neg) triples
    """
    print_message("#> Loading triples...")

    triples = []

    with open(path) as f:
        for line_idx, line in enumerate(f):
            # qid, pos, neg = json.loads(line)
            qid, pos, *negs = line.strip().split("\t")
            negs = [int(neg) for neg in negs]
            triples.append((int(qid), int(pos), *negs))

    return triples


def load_qrel(path):
    """
    returns: list of (qid, pid) pairs
    """
    print_message(f"#> Loading qrel from {path}")

    qrel = []

    with open(path) as f:
        for line in file_tqdm(f):
            # qid, pos, neg = json.loads(line)
            qid, _, pid, *_ = line.strip().split("\t")
            qrel.append((int(qid), int(pid)))

    return qrel


def load_candidates(path):
    """
    returns: dict of {qid: [pids]}
    """
    if path is None:
        return None

    if not os.path.exists(path):
        print_message(f"#> Candidates file not found at {path}.")
        return None

    print_message(f"#> Loading candidates from {path}")

    candidates = json.load(
        open(path, "r", encoding="utf-8"),
        object_pairs_hook=lambda x: {int(k): v for k, v in x},
    )

    return candidates


nltk_stopwords = set(stopwords.words("english"))


def clean(
    corpus,
    rm_punctuation=False,
    rm_stopwords=None,
    lower=False,
    is_corpus_list=False,
    cache_dir="cache",
    data_name="corpus",
):
    """
    corpus: dict of {key: text} or list of {key: text}
    Clean text by removing HTML tags, fixing unicode, removing extra spaces
    """
    cache_file = generate_cache_file_name(
        rm_punctuation, rm_stopwords, lower, is_corpus_list, cache_dir, data_name
    )
    cleaning_options = (
        f"rm_punctuation={rm_punctuation} |"
        f"rm_stopwords={rm_stopwords} | lower={lower} | remove html tags"
    )
    print_message("#> Cleaning corpus with options:", cleaning_options)

    if os.path.exists(cache_file):
        print_message("#> Loading cleaned corpus from cache.")
        with open(cache_file, "r", encoding="utf-8") as cache_file:
            if is_corpus_list:
                cleaned_corpus = json.load(cache_file)
            else:
                cleaned_corpus = json.load(
                    cache_file, object_pairs_hook=lambda x: {int(k): v for k, v in x}
                )  # convert str keys to int keys
    else:
        cleaned_corpus = _clean_corpus(
            corpus, rm_punctuation, rm_stopwords, lower, is_corpus_list
        )
        print_message("#> Saving cleaned corpus to cache.")
        with open(cache_file, "w", encoding="utf-8") as cache_file:
            json.dump(cleaned_corpus, cache_file, indent=2, ensure_ascii=False)

    # Print one example before and after cleaning
    if is_corpus_list:
        print("#> Example before cleaning:", corpus[0]["passage"])
        print("#> Example after cleaning:", cleaned_corpus[0]["passage"])
    else:
        first_key = next(iter(corpus))
        print("#> Example before cleaning:", corpus[first_key])
        print("#> Example after cleaning:", cleaned_corpus[first_key])

    return cleaned_corpus


def generate_cache_file_name(
    rm_punctuation, rm_stopwords, lower, is_corpus_list, cache_dir, data_name
):
    cache_file_name = f"{data_name}_cache"
    if rm_punctuation:
        cache_file_name += "_rp"
    if rm_stopwords:
        cache_file_name += "_rs"
    if lower:
        cache_file_name += "_l"
    if is_corpus_list:
        cache_file_name += "_list"
    cache_file_name += ".json"
    cache_file_path = os.path.join(cache_dir, cache_file_name)
    create_directory(cache_dir)
    return cache_file_path


def _clean_corpus(corpus, rm_punctuation, rm_stopwords, lower, is_corpus_list):
    if is_corpus_list:
        cleaned_corpus = []
        for d in tqdm.tqdm(corpus):
            cleaned_passage = _clean(
                d["passage"],
                rm_punctuation=rm_punctuation,
                rm_stopwords=rm_stopwords,
                lower=lower,
            )
            cleaned_corpus.append({"pid": d["pid"], "passage": cleaned_passage})
    else:
        cleaned_corpus = {}
        for key in tqdm.tqdm(corpus):
            cleaned_text = _clean(
                corpus[int(key)],
                rm_punctuation=rm_punctuation,
                rm_stopwords=rm_stopwords,
                lower=lower,
            )
            cleaned_corpus[key] = cleaned_text

    return cleaned_corpus


def _clean(text, rm_punctuation=False, rm_stopwords=None, lower=False):
    """
    Clean text by removing HTML tags, fixing unicode, removing extra spaces
    """
    text = re.sub(r"\s+", " ", text)
    text = ftfy.fix_text(text).strip()

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    if lower:
        text = text.lower()
    if rm_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    if rm_stopwords:
        text = " ".join([w for w in text.split() if w not in nltk_stopwords])

    return text


def do_span(
    query, passage, n_samples=10, ngram=3, temperature=1.0, min_length=10, max_length=10
):
    """
    split passage into spans of length (min_length, max_length)
    """

    def span_iterator(tokens, ngram=3, banned=nltk_stopwords):
        for i in range(len(tokens)):
            if tokens[i] not in banned:
                yield (i, i + ngram)

    query = query.split()
    query_lower = [w.lower() for w in query]
    passage = passage.split()
    passage_lower = [w.lower() for w in passage]

    matches = defaultdict(int)

    for i1, _ in enumerate(query_lower):
        j1 = i1 + ngram
        str_1 = " ".join(query_lower[i1:j1])

        for i2, j2 in span_iterator(passage_lower, ngram=ngram):
            str_2 = " ".join(passage_lower[i2:j2])
            ratio = fuzz.ratio(str_1, str_2) / 100.0
            matches[i2] += ratio

    if not matches:
        indices = [0]

    else:
        indices, weights = zip(*sorted(matches.items(), key=lambda x: -(x[1])))
        weights = list(weights)
        sum_weights = float(sum([0] + weights))
        if sum_weights == 0.0 or not weights:
            indices = [0]
            weights = [1.0]
        else:
            weights = [math.exp(float(w) / temperature) for w in weights]
            Z = sum(weights)
            weights = [w / Z for w in weights]

        indices = random.choices(indices, weights=weights, k=n_samples)

    spans = []
    for i in indices:
        subspan_size = random.randint(min_length, max_length)
        span = " ".join(passage[i : i + subspan_size])
        spans.append(span)

    return spans


# def batch(file, bsize):
#     while True:
#         L = [json.loads(file.readline()) for _ in range(bsize)]
#         yield L
#     return


def f7(seq):
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """

    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        L = group[offset : offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    Credit: derek73 @ https://stackoverflow.com/questions/2352181
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten(L):
    return [x for y in L for x in y]


def zipstar(L, lazy=False):
    """
    A much faster A, B, C = zip(*[(a, b, c), (a, b, c), ...])
    May return lists or tuples.
    """

    if len(L) == 0:
        return L

    width = len(L[0])

    if width < 100:
        return [[elem[idx] for elem in L] for idx in range(width)]

    L = zip(*L)

    return L if lazy else list(L)


def zip_first(L1, L2):
    length = len(L1) if type(L1) in [tuple, list] else None

    L3 = list(zip(L1, L2))

    assert length in [None, len(L3)], "zip_first() failure: length differs!"

    return L3


def int_or_float(val):
    if "." in val:
        return float(val)

    return int(val)


def load_ranking(path, types=None, lazy=False):
    print_message(f"#> Loading the ranked lists from {path} ..")

    try:
        lists = torch.load(path)
        lists = zipstar([l.tolist() for l in tqdm.tqdm(lists)], lazy=lazy)
    except:
        if types is None:
            types = itertools.cycle([int_or_float])

        with open(path) as f:
            lists = [
                [typ(x) for typ, x in zip_first(types, line.strip().split("\t"))]
                for line in file_tqdm(f)
            ]

    return lists


def save_ranking(ranking, path):
    lists = zipstar(ranking)
    lists = [torch.tensor(l) for l in lists]

    torch.save(lists, path)

    return lists


def groupby_first_item(lst):
    groups = defaultdict(list)

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest
        groups[first].append(tuple(rest))

    return groups


def process_grouped_by_first_item(lst):
    """
    Requires items in list to already be grouped by first item.
    """

    groups = defaultdict(list)

    started = False
    last_group = None

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest

        if started and first != last_group:
            yield (last_group, groups[last_group])
            assert (
                first not in groups
            ), f"{first} seen earlier --- violates precondition."

        groups[first].append(rest)

        last_group = first
        started = True

    return groups


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
        Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def exist_file_with_prefix(path, prefix):
    for filename in os.listdir(path):
        if filename.startswith(prefix):
            return True
    return False


class Timer:
    def __init__(self, description=""):
        self.description = description

    def __enter__(self):
        print_message("#> Start - ", self.description)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        print_message(
            f"#> {self.description} - Time elapsed: {elapsed_time:.4f} seconds"
        )
