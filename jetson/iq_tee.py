#!/usr/bin/env python3
"""
iq_tee.py — IQ stream tee for parallel GNSS-SDR operation

This module provides IQTee, a writer that mirrors raw IQ samples to a
named FIFO pipe while demo_inference.py continues its normal classification
pipeline. GNSS-SDR reads from the other end of the pipe as a
File_Signal_Source (see gnss_sdr_fifo.conf).

Data format written to the FIFO:
    Interleaved int16: [I0, Q0, I1, Q1, ...]
    This matches GNSS-SDR's "ishort" item_type and is the same format
    as OAKBAT .bin files and the PlutoSDR's raw ADC output.

Architecture:
    PlutoSDR ─→ demo_inference.py ─┬─→ GPU preprocessing ─→ TRT ─→ class
                                   │
                                   └─→ IQTee ─→ FIFO ─→ GNSS-SDR ─→ PVT

Thread safety:
    The FIFO write is performed in a background thread to prevent the
    blocking write() call from stalling the classifier's real-time loop.
    A bounded queue (default depth = 5 windows) absorbs timing jitter.
    If the queue is full (GNSS-SDR is too slow or stuck), the oldest
    window is dropped and a warning is printed — the classifier never
    blocks waiting for GNSS-SDR.

Usage in demo_inference.py:
    from iq_tee import IQTee

    tee = IQTee("/tmp/gnss_iq_fifo")   # or None if --fifo not specified
    ...
    # Inside the capture loop, after receiving a 20 ms IQ window:
    if tee is not None:
        tee.write(iq_int16)            # raw int16 array before float conversion
    ...
    # On exit:
    if tee is not None:
        tee.close()
"""

import os
import sys
import threading
import queue
import numpy as np
from typing import Optional


class IQTee:
    """
    Non-blocking IQ sample writer to a named FIFO pipe.

    The write() method enqueues int16 IQ data for a background thread
    to flush to the pipe. If the queue is full, the oldest sample is
    dropped to keep the classifier running at full speed.

    Parameters
    ----------
    fifo_path : str
        Path to the named FIFO (must already exist, created by mkfifo).
    max_queue_depth : int
        Maximum number of IQ windows to buffer. Each window is 100,000
        complex samples = 400 KB as int16 pairs. At depth 5, this uses
        ~2 MB — negligible on the Jetson's 4 GB.
    """

    def __init__(self, fifo_path: str, max_queue_depth: int = 5):
        self.fifo_path = fifo_path
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_depth)
        self._stop_event = threading.Event()
        self._fd: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0
        self._written = 0

        # Validate FIFO exists
        if not os.path.exists(fifo_path):
            raise FileNotFoundError(
                f"FIFO not found: {fifo_path}\n"
                f"Create it with: mkfifo {fifo_path}"
            )

        # Check it's actually a FIFO (not a regular file)
        import stat
        mode = os.stat(fifo_path).st_mode
        if not stat.S_ISFIFO(mode):
            raise ValueError(
                f"{fifo_path} exists but is not a FIFO (named pipe).\n"
                f"Remove it and recreate: rm {fifo_path} && mkfifo {fifo_path}"
            )

        # Start the writer thread. It will block on open() until a reader
        # (GNSS-SDR) opens the other end. This is fine because it's in a
        # background thread — the classifier can start capturing immediately.
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="iq_tee_writer",
            daemon=True,
        )
        self._thread.start()
        print(f"[IQTee] Writer thread started → {fifo_path}")
        print(f"[IQTee] Waiting for reader (GNSS-SDR) to open the pipe...")

    def write(self, iq_int16: np.ndarray) -> None:
        """
        Enqueue an IQ window for writing to the FIFO.

        Parameters
        ----------
        iq_int16 : np.ndarray
            Raw interleaved int16 array of shape (2*N,) where N is the
            number of complex samples. This is the format directly from
            the PlutoSDR's DMA buffer before float conversion.
            Alternatively, shape (N, 2) with dtype int16 is also accepted
            and will be flattened.
        """
        if self._stop_event.is_set():
            return

        # Ensure contiguous int16 for the write
        data = np.ascontiguousarray(iq_int16.ravel().astype(np.int16))

        try:
            self._queue.put_nowait(data.tobytes())
        except queue.Full:
            # Drop the oldest to make room — classifier never blocks
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(data.tobytes())
            except queue.Full:
                pass
            self._dropped += 1
            if self._dropped % 50 == 1:
                print(f"[IQTee] WARNING: queue full, dropped {self._dropped} "
                      f"windows (GNSS-SDR may be lagging)")

    def close(self) -> None:
        """Flush remaining data and close the FIFO."""
        self._stop_event.set()

        # Put a sentinel to unblock the writer if it's waiting on get()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        print(f"[IQTee] Closed. Written: {self._written} windows, "
              f"dropped: {self._dropped}")

    def _writer_loop(self) -> None:
        """Background thread: opens FIFO and writes queued data."""
        try:
            # os.open with O_WRONLY will block until a reader opens the FIFO.
            # This is intentional — GNSS-SDR must be started first (or
            # concurrently via launch_parallel.sh).
            self._fd = os.open(self.fifo_path, os.O_WRONLY)
            print(f"[IQTee] FIFO connected (reader attached)")
        except OSError as e:
            print(f"[IQTee] ERROR: could not open FIFO: {e}")
            return

        try:
            while not self._stop_event.is_set():
                try:
                    data = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if data is None:  # sentinel
                    break

                try:
                    os.write(self._fd, data)
                    self._written += 1
                except BrokenPipeError:
                    print("[IQTee] Reader (GNSS-SDR) closed the pipe.")
                    break
                except OSError as e:
                    print(f"[IQTee] Write error: {e}")
                    break
        finally:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
