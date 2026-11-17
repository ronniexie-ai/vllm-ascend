"""DMA copy backend for NPU<->CPU block transfers.

Mirrors :class:`vllm.v1.simple_kv_offload.copy_backend.DmaCopyBackend`
but routes batched memcpy through ``torch.ops._C_ascend.swap_blocks_batch``
and uses ``torch.npu`` streams/events.
"""

from __future__ import annotations

import queue
import threading

import torch

from vllm.logger import logger

from vllm_ascend.simple_kv_offload.npu_mem_ops import (
    DIRECTION_D2H,
    DIRECTION_H2D,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)


class NPUDmaCopyBackend:
    """``aclrtMemcpyBatchAsync`` copy backend running on a worker thread.

    Two pre-built ``BatchMemcpyParams`` are cached (load=H2D, store=D2H).
    Submitted jobs are dispatched in FIFO order to a single worker
    thread; each job issues its copies on a dedicated NPU stream and
    records an Event the main thread can poll without synchronizing
    the device.
    """

    _SHUTDOWN_TIMEOUT = 5.0
    _SUBMISSION_TIMEOUT = 30.0
    # Failed device cleanup must not drop descriptors still used by DMA.
    # Retain these backends until cleanup succeeds or the process exits.
    _shutdown_pending: set[NPUDmaCopyBackend] = set()

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.npu.Stream | None = None
        self._store_stream: torch.npu.Stream | None = None
        self._device: torch.device | None = None
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._submission_lock = threading.Lock()
        self._shutdown: bool = False
        self._shutdown_thread: threading.Thread | None = None
        self._copy_error: Exception | None = None
        # Hold descriptor tensors until their matching copy event is observed
        # complete. A fixed-size ring can release descriptors during a slow DMA.
        self._keep_alive: dict[tuple[bool, int], object] = {}
        self._keep_alive_lock = threading.Lock()

    def init(
        self,
        npu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.npu.Stream,
        store_stream: torch.npu.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._device = device
        # Stores go NPU->CPU (D2H), loads go CPU->NPU (H2D).
        self._store_params = build_params(npu_caches, cpu_caches, DIRECTION_D2H)
        self._load_params = build_params(cpu_caches, npu_caches, DIRECTION_H2D)

        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            name="npu-kv-offload-copy",
            daemon=True,
        )
        self._thread.start()
        logger.info("KV offload descriptor lifetime and submission-drain fence enabled")

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.npu.Event]],
        wait_event: torch.npu.Event | None = None,
    ) -> None:
        if self._shutdown:
            raise RuntimeError("KV copy backend is shut down")
        self.check_error()
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        # Diagnostic ordering: loads must also follow already-enqueued compute.
        # Waiting only for copy completion does not prevent an earlier writer
        # from racing with the load destination on another stream.
        if wait_event is None:
            wait_event = torch.npu.Event()
            wait_event.record(torch.npu.current_stream())
        # Serialize admission with the stop marker; no job may follow it.
        with self._submission_lock:
            if self._shutdown:
                raise RuntimeError("KV copy backend is shut down")
            self.check_error()
            self._queue.put((src_blocks, dst_blocks, params, is_store, event_idx, events_list, wait_event))
        # Diagnostic arm: complete this DMA before model/draft work can proceed.
        # This deliberately trades overlap for deterministic copy ordering.
        self.drain_submissions()
        for submitted_idx, event in events_list:
            if submitted_idx == event_idx:
                event.synchronize()
                break
        else:
            raise RuntimeError("KV copy event was not published after queue drain")

    def check_error(self) -> None:
        if self._copy_error is not None:
            raise RuntimeError("NPU KV copy worker failed") from self._copy_error

    def drain_submissions(self) -> None:
        """Publish events for all queued jobs before the caller walks them.

        The model worker is the sole producer. Queue completion means the copy
        thread has recorded and published each event, not that DMA is complete.
        The caller still synchronizes the events before allowing page reuse.
        """
        if self._queue is not None:
            # Queue.join has no timeout. Use its completion condition with
            # the same unfinished-task predicate, without releasing DMA pins.
            with self._queue.all_tasks_done:
                drained = self._queue.all_tasks_done.wait_for(
                    lambda: self._queue.unfinished_tasks == 0,
                    timeout=self._SUBMISSION_TIMEOUT,
                )
            if not drained:
                self.check_error()
                raise TimeoutError("NPU KV copy submission drain timed out")
        self.check_error()

    def shutdown(self) -> None:
        with self._submission_lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._shutdown_pending.add(self)
            if self._queue is not None:
                self._queue.put(None)
        # Both host submission and device synchronization can stall. Bound
        # the caller's wait while a daemon retains and safely drains resources.
        try:
            cleanup_thread = threading.Thread(
                target=self._finish_shutdown,
                name="npu-kv-offload-shutdown",
                daemon=True,
            )
            cleanup_thread.start()
        except RuntimeError:
            # Interpreter teardown may forbid new threads. Do not fall back
            # to unbounded synchronous cleanup or free live DMA resources.
            logger.exception(
                "NPU KV copy shutdown thread could not start; "
                "retaining DMA resources until process exit"
            )
            return
        self._shutdown_thread = cleanup_thread
        self._shutdown_thread.join(timeout=self._SHUTDOWN_TIMEOUT)
        if self._shutdown_thread.is_alive():
            logger.warning(
                "NPU KV copy shutdown timed out after %.1fs; "
                "retaining DMA resources until cleanup completes",
                self._SHUTDOWN_TIMEOUT,
            )

    def _finish_shutdown(self) -> None:
        try:
            if self._thread is not None:
                self._thread.join()
            if self._device is not None:
                torch.npu.set_device(self._device)
            # Host completion alone does not establish DMA completion.
            for stream in (self._load_stream, self._store_stream):
                if stream is not None:
                    stream.synchronize()
            with self._keep_alive_lock:
                self._keep_alive.clear()
        except Exception:
            logger.exception(
                "NPU KV copy shutdown cleanup failed; retaining DMA resources "
                "until process exit"
            )
            return
        self._shutdown_pending.discard(self)
        if self._copy_error is not None:
            logger.error("NPU KV copy worker failed before shutdown: %r", self._copy_error)

    def release_completed(self, is_store: bool, event_idx: int) -> None:
        """Release descriptor tensors after the corresponding event completes."""
        with self._keep_alive_lock:
            self._keep_alive.pop((bool(is_store), int(event_idx)), None)

    # ------------------------------------------------------------------
    # Worker thread main loop
    # ------------------------------------------------------------------
    def _copy_loop(self) -> None:
        # launch_copy records dependencies on the caller's compute stream.
        # Consume them on the selected DMA stream before accessing KV pages.
        # Allocator pins prevent reuse; stream events order device accesses.
        assert self._device is not None
        assert self._queue is not None
        assert self._load_stream is not None
        assert self._store_stream is not None
        try:
            torch.npu.set_device(self._device)
        except Exception as exc:
            self._copy_error = exc

        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                # Keep draining after an error so a flush can report it rather
                # than deadlocking on jobs left in the host queue.
                if self._copy_error is not None:
                    continue
                (
                    src_blocks,
                    dst_blocks,
                    params,
                    is_store,
                    event_idx,
                    events_list,
                    wait_event,
                ) = item

                stream = self._store_stream if is_store else self._load_stream
                with torch.npu.stream(stream):
                    if wait_event is not None:
                        stream.wait_event(wait_event)
                    keep_alive = copy_blocks(src_blocks, dst_blocks, params)
                    if keep_alive is not None:
                        with self._keep_alive_lock:
                            self._keep_alive[(bool(is_store), int(event_idx))] = keep_alive
                    event = torch.npu.Event()
                    event.record(stream)
                events_list.append((event_idx, event))
            except Exception as exc:
                self._copy_error = exc
            finally:
                self._queue.task_done()
