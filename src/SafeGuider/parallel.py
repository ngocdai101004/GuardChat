"""Multi-process fan-out for the SafeGuider beam search.

Measured on a 16-core box: one search process pins **one** core at 100%
for the whole run while the GPU idles at ~49% and holds 738 MiB. Roughly
90% of the wall clock is Python — building the ~570 candidate strings
each depth expands, and tokenising them — not the encoder. So the lever
is more processes, not a bigger ``--batch-size`` and not a bigger GPU.

Threads cannot do it: the hot loop is pure Python bytecode, so the GIL
serialises them. Each worker is a separate interpreter with its own
model copy.

Cost per worker: ~740 MiB of weights plus a CUDA context (~300-500 MiB),
so about **1.2 GiB of GPU memory each** - that, not the core count, is
usually what caps the fan-out. :func:`suggest_workers` reads both.

Samples differ in cost by more than 30x (a prompt that qualifies at
depth 2 versus one that stalls out at depth 25), so work is handed out
one sample at a time and consumed as it completes. Chunking would leave
workers idle behind a single slow neighbour.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Sequence

from src.utils import GuardChatSample, RewriteRecord, normalise_rewrite_kind

from .rewrite import (
    DEFAULT_ENCODER_MODEL,
    DEFAULT_WEIGHTS,
    RewritePipeline,
)


# Weights + CUDA context per worker process. Deliberately generous - the
# failure mode it guards against is a mid-run CUDA OOM that takes an
# entire worker's queue with it.
GPU_MIB_PER_WORKER = 1200


@dataclass
class WorkerConfig:
    """Everything a worker needs to rebuild the pipeline from scratch.

    Workers are spawned, not forked (CUDA does not survive a fork), so
    nothing is inherited: this must be self-contained and picklable.
    """

    weights: str = DEFAULT_WEIGHTS
    encoder_model: str = DEFAULT_ENCODER_MODEL
    device: Optional[str] = None
    beam_width: int = 6
    max_depth: int = 25
    safety_threshold: float = 0.80
    similarity_floor: float = 0.1
    batch_size: int = 64
    patience: int = 0
    gate: str = "recognizer"


# One pipeline per worker process, built once in the initialiser and
# reused for every sample that process is handed.
_PIPE: Optional[RewritePipeline] = None


def _init_worker(cfg_dict: dict) -> None:
    global _PIPE
    _PIPE = RewritePipeline.from_weights(**cfg_dict)


def _rewrite_one(args) -> RewriteRecord:
    sample, kind = args
    assert _PIPE is not None, "worker initialiser did not run"
    return _PIPE.rewrite_sample(sample, kind=kind)


def suggest_workers(gpu_mib: Optional[int] = None) -> int:
    """A fan-out that fits both the CPU and the GPU.

    Leaves two cores for the parent process and the OS. Returns 1 when
    anything is unknown, since a wrong guess here costs a crashed run.
    """
    cores = os.cpu_count() or 2
    by_cpu = max(1, cores - 2)

    if gpu_mib is None:
        try:
            import torch
            if not torch.cuda.is_available():
                return by_cpu          # CPU-only: memory is not the cap
            free, _total = torch.cuda.mem_get_info()
            gpu_mib = int(free / 2**20)
        except Exception:              # noqa: BLE001 - advisory only
            return 1

    by_gpu = max(1, int(gpu_mib) // GPU_MIB_PER_WORKER)
    return max(1, min(by_cpu, by_gpu))


class ParallelRewriter:
    """Runs :class:`RewritePipeline` across several worker processes.

    Exposes the same ``rewrite_samples(samples, kind, on_result)`` shape
    as the single-process pipeline, so :func:`src.utils.rewrite_kind`
    drives either one without knowing the difference - and checkpointing
    keeps working, because ``on_result`` still fires once per finished
    sample, in the parent.
    """

    def __init__(self, config: WorkerConfig, workers: int) -> None:
        self.config = config
        self.workers = max(1, int(workers))

    @property
    def model_name(self) -> str:
        return "SafeGuider-beam-search"

    def rewrite_samples(
        self,
        samples: Sequence[GuardChatSample],
        kind: str = "prompt",
        on_result: Optional[Callable[[RewriteRecord], None]] = None,
        progress: bool = True,
    ) -> List[RewriteRecord]:
        kind = normalise_rewrite_kind(kind)
        if not samples:
            return []

        import multiprocessing as mp

        cfg = asdict(self.config)
        bar = self._progress_bar(len(samples), kind) if progress else None
        out: List[RewriteRecord] = []

        # spawn, not fork: a forked child inherits a CUDA context it
        # cannot use, and the first kernel launch dies.
        ctx = mp.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(cfg,),
        )
        try:
            futures = [executor.submit(_rewrite_one, (s, kind)) for s in samples]
            # as_completed, not map: costs vary by 30x, so waiting for
            # results in submission order would stall the checkpoint
            # behind whichever sample happens to be slowest.
            for fut in as_completed(futures):
                rec = fut.result()
                if on_result is not None:
                    on_result(rec)
                out.append(rec)
                if bar is not None:
                    bar.update(1)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            if bar is not None:
                bar.close()
        return out

    @staticmethod
    def _progress_bar(total: int, kind: str) -> Any:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return None
        return tqdm(total=total, desc=f"SafeGuider[{kind}]")


__all__ = [
    "GPU_MIB_PER_WORKER",
    "ParallelRewriter",
    "WorkerConfig",
    "suggest_workers",
]
