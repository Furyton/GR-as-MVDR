"""
    Heavily based on: https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_1bn.py
"""


import os
import sys
import time
import faiss
import torch
import numpy as np
from utils import print_message
from multiprocessing import Pool
import faiss.contrib.torch_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FaissIndexGPU:
    def __init__(self):
        self.ngpu = faiss.get_num_gpus()

        if self.ngpu == 0:
            return

        self.tempmem = 1 << 33
        self.max_add_per_gpu = 1 << 25
        self.max_add = self.max_add_per_gpu * self.ngpu
        self.add_batch_size = 65536

        self.gpu_resources = self._prepare_gpu_resources()

    def _prepare_gpu_resources(self):
        print_message(f"Preparing resources for {self.ngpu} GPUs.")

        gpu_resources = []

        for _ in range(self.ngpu):
            res = faiss.StandardGpuResources()
            if self.tempmem >= 0:
                res.setTempMemory(self.tempmem)
            gpu_resources.append(res)

        return gpu_resources

    def _make_vres_vdev(self):
        """
        return vectors of device ids and resources useful for gpu_multiple
        """

        assert self.ngpu > 0

        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()

        for i in range(self.ngpu):
            vdev.push_back(i)
            vres.push_back(self.gpu_resources[i])

        return vres, vdev

    def training_initialize(self, index, quantizer):
        """
        The index and quantizer should be owned by caller.
        """

        assert self.ngpu > 0

        s = time.time()
        self.index_ivf = faiss.extract_index_ivf(index)
        self.clustering_index = faiss.index_cpu_to_all_gpus(quantizer)
        self.index_ivf.clustering_index = self.clustering_index
        print(time.time() - s)

    def training_finalize(self):
        assert self.ngpu > 0

        s = time.time()
        self.index_ivf.clustering_index = faiss.index_gpu_to_cpu(
            self.index_ivf.clustering_index
        )
        print(time.time() - s)

    def adding_initialize(self, index):
        """
        The index should be owned by caller.
        """

        assert self.ngpu > 0

        self.co = faiss.GpuMultipleClonerOptions()
        self.co.useFloat16 = True
        self.co.useFloat16CoarseQuantizer = False
        self.co.usePrecomputed = False
        self.co.indicesOptions = faiss.INDICES_CPU
        self.co.verbose = True
        self.co.reserveVecs = self.max_add
        self.co.shard = True
        assert self.co.shard_type in (0, 1, 2)

        self.vres, self.vdev = self._make_vres_vdev()
        self.gpu_index = faiss.index_cpu_to_gpu_multiple(
            self.vres, self.vdev, index, self.co
        )

    def add(self, index, data, offset):
        assert self.ngpu > 0

        t0 = time.time()
        nb = data.shape[0]

        for i0 in range(0, nb, self.add_batch_size):
            i1 = min(i0 + self.add_batch_size, nb)
            xs = data[i0:i1]

            self.gpu_index.add_with_ids(xs, np.arange(offset + i0, offset + i1))

            if self.max_add > 0 and self.gpu_index.ntotal > self.max_add:
                self._flush_to_cpu(index, nb, offset)

            print("\r%d/%d (%.3f s)  " % (i0, nb, time.time() - t0), end=" ")
            sys.stdout.flush()

        if self.gpu_index.ntotal > 0:
            self._flush_to_cpu(index, nb, offset)

        assert index.ntotal == offset + nb, (index.ntotal, offset + nb, offset, nb)
        print(
            f"add(.) time: %.3f s \t\t--\t\t index.ntotal = {index.ntotal}"
            % (time.time() - t0)
        )

    def _flush_to_cpu(self, index, nb, offset):
        print("Flush indexes to CPU")

        for i in range(self.ngpu):
            index_src_gpu = faiss.downcast_index(
                self.gpu_index if self.ngpu == 1 else self.gpu_index.at(i)
            )
            index_src = faiss.index_gpu_to_cpu(index_src_gpu)

            index_src.copy_subset_to(index, 0, offset, offset + nb)
            index_src_gpu.reset()
            index_src_gpu.reserveMemory(self.max_add)

        if self.ngpu > 1:
            try:
                self.gpu_index.sync_with_shard_indexes()
            except:
                self.gpu_index.syncWithSubIndexes()


class FaissIndex:
    def __init__(self, dim, partitions):
        self.dim = dim
        self.partitions = partitions

        self.gpu = FaissIndexGPU()
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatIP(self.dim)  # faiss.IndexHNSWFlat(dim, 32)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)

        return quantizer, index

    def train(self, train_data):
        print_message(f"#> Training now (using {self.gpu.ngpu} GPUs)...")

        if self.gpu.ngpu > 0:
            self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

        if self.gpu.ngpu > 0:
            self.gpu.training_finalize()

    def add(self, data):
        print_message(f"Add data with shape {data.shape} (offset = {self.offset})..")

        if self.gpu.ngpu > 0 and self.offset == 0:
            self.gpu.adding_initialize(self.index)

        if self.gpu.ngpu > 0:
            self.gpu.add(self.index, data, self.offset)
        else:
            self.index.add(data)

        self.offset += data.shape[0]

    def save(self, output_path):
        print_message(f"Writing index to {output_path} ...")

        self.index.nprobe = 10  # just a default
        faiss.write_index(self.index, output_path)


class FaissRetrieveIndex:
    def __init__(self, faiss_index_path, emb2pid, nprobe):
        print_message("#> Loading the FAISS index from", faiss_index_path, "..")

        res = faiss.StandardGpuResources()
        faiss_index = faiss.read_index(faiss_index_path)
        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        print_message(
            "#> nlist of faiss index:",
            self.faiss_index.nlist,
            " nprobe of faiss index",
            self.faiss_index.nprobe,
        )
        self.faiss_index.nprobe = nprobe
        assert self.faiss_index.nprobe <= self.faiss_index.nlist, self.faiss_index.nlist

        # print_message("#> Building the emb2pid mapping..")
        # all_doclens, emb2pid = load_meta(index_path)

        # total_num_embeddings = sum(all_doclens.values())
        # self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)
        self.emb2pid = torch.tensor(emb2pid, dtype=torch.int, device=DEVICE)
        # assert len(self.emb2pid) == total_num_embeddings

        print_message("len(self.emb2pid) =", len(self.emb2pid))

        self.parallel_pool = Pool(16)

    def retrieve(self, faiss_depth, Q, verbose=False):
        embedding_ids = self.queries_to_embedding_ids(faiss_depth, Q, verbose=verbose)
        pids = self.embedding_ids_to_pids(embedding_ids, verbose=verbose)

        return pids

    def queries_to_embedding_ids(self, faiss_depth, Q, verbose=True):
        # Flatten into a matrix for the faiss search.
        num_queries, embeddings_per_query, dim = Q.size()
        Q_faiss = Q.view(num_queries * embeddings_per_query, dim)

        # Search in large batches with faiss.
        print_message(
            "#> Search in batches with faiss. \t\t",
            f"Q.size() = {Q.size()}, Q_faiss.size() = {Q_faiss.size()}",
            condition=verbose,
        )

        embeddings_ids = []
        faiss_bsize = embeddings_per_query * 5000
        for offset in range(0, Q_faiss.size(0), faiss_bsize):
            endpos = min(offset + faiss_bsize, Q_faiss.size(0))

            print_message(
                "#> Searching from {} to {}...".format(offset, endpos),
                condition=verbose,
            )

            some_Q_faiss = Q_faiss[offset:endpos]
            # assert len(self.faiss_index.nlist) >= self.faiss_index.nprobe, len(some_Q_faiss)
            _, some_embedding_ids = self.faiss_index.search(some_Q_faiss, faiss_depth)
            embeddings_ids.append(some_embedding_ids)

        embedding_ids = torch.cat(embeddings_ids)

        # Reshape to (number of queries, non-unique embedding IDs per query)
        embedding_ids = embedding_ids.view(
            num_queries, embeddings_per_query * embedding_ids.size(1)
        )

        return embedding_ids

    def embedding_ids_to_pids(self, embedding_ids, verbose=True):
        # Find unique PIDs per query.
        print_message("#> Lookup the PIDs..", condition=verbose)
        all_pids = self.emb2pid[embedding_ids]

        print_message(
            f"#> Converting to a list [shape = {all_pids.size()}]..", condition=verbose
        )
        all_pids = all_pids.cpu().tolist()

        print_message(
            "#> Removing duplicates (in parallel if large enough)..", condition=verbose
        )

        if len(all_pids) > 1000:
            all_pids = list(self.parallel_pool.map(uniq, all_pids))
        else:
            all_pids = list(map(uniq, all_pids))

        print_message("#> Done with embedding_ids_to_pids().", condition=verbose)

        return all_pids


def uniq(l):
    return list(set(l))


def flatten(L):
    return [x for y in L for x in y]
