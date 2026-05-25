"""
Functions for training diagnostics, including timing and memory usage (RAM and VRAM)
logging.
"""
import time

try:
    import psutil
except ImportError:
    psutil = None

import torch


class TrainingDiagnostics:
    """Timing and memory diagnostics shared by training modules."""

    def _reset_timing_counters(self) -> None:
        self._timing_steps = 0
        self._timing_forward_sum = 0.0
        self._timing_main_loss_sum = 0.0
        self._timing_aux_loss_sum = 0.0
        self._timing_logging_sum = 0.0
        self._timing_step_total_sum = 0.0

    def _sync_for_timing(self) -> None:
        if self.profile_timing_cuda_sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _mark_time(self) -> float:
        self._sync_for_timing()
        return time.perf_counter()

    def _elapsed_since(self, start_ts: float) -> float:
        self._sync_for_timing()
        return time.perf_counter() - start_ts

    def _log_timing_usage(self, tb_logging: bool = False) -> None:
        """
        Log timing usage for different stages of the training step.
        Prints to console by default, and also logs to TensorBoard if tb_logging=True.

        Note: Called at the end of each training epoch to track timing trends
        over time.
        """
        
        if not getattr(self, "profile_timing_enabled", True):
            return
        if getattr(self, "_timing_steps", 0) <= 0:
            return

        steps = float(getattr(self, "_timing_steps", 0))
        forward_avg = getattr(self, "_timing_forward_sum", 0.0) / steps
        main_loss_avg = getattr(self, "_timing_main_loss_sum", 0.0) / steps
        aux_loss_avg = getattr(self, "_timing_aux_loss_sum", 0.0) / steps
        logging_avg = getattr(self, "_timing_logging_sum", 0.0) / steps
        step_total_avg = getattr(self, "_timing_step_total_sum", 0.0) / steps

        total_denom = max(step_total_avg, 1e-12)
        forward_pct = 100.0 * (forward_avg / total_denom)
        main_loss_pct = 100.0 * (main_loss_avg / total_denom)
        aux_loss_pct = 100.0 * (aux_loss_avg / total_denom)
        logging_pct = 100.0 * (logging_avg / total_denom)

        if tb_logging:
            self.log("timing_step_forward_s", forward_avg, on_step=False, on_epoch=True, logger=True)
            self.log("timing_step_main_loss_s", main_loss_avg, on_step=False, on_epoch=True, logger=True)
            self.log("timing_step_aux_loss_s", aux_loss_avg, on_step=False, on_epoch=True, logger=True)
            self.log("timing_step_logging_s", logging_avg, on_step=False, on_epoch=True, logger=True)
            self.log("timing_step_total_s", step_total_avg, on_step=False, on_epoch=True, logger=True)

            self.log("timing_pct_forward", forward_pct, on_step=False, on_epoch=True, logger=True)
            self.log("timing_pct_main_loss", main_loss_pct, on_step=False, on_epoch=True, logger=True)
            self.log("timing_pct_aux_loss", aux_loss_pct, on_step=False, on_epoch=True, logger=True)
            self.log("timing_pct_logging", logging_pct, on_step=False, on_epoch=True, logger=True)

        if self.profile_timing_print:
            print(
                f"Timing epoch {self.current_epoch + 1}: "
                f"step_total={step_total_avg:.3f}s, "
                f"forward={forward_avg:.3f}s ({forward_pct:.1f}%), "
                f"main_loss={main_loss_avg:.3f}s ({main_loss_pct:.1f}%), "
                f"aux_loss={aux_loss_avg:.3f}s ({aux_loss_pct:.1f}%), "
                f"logging={logging_avg:.3f}s ({logging_pct:.1f}%)"
            )

    def _log_memory_usage(self, tb_logging: bool = False) -> None:
        """
        Log system RAM and GPU VRAM usage.
        By default, prints to console. If tb_logging=True, also logs to TensorBoard.

        Note: Called at the end of each training epoch to track memory usage trends
        over time.

        Metric types:
        - RAM RSS: Physical memory used by the process (excludes swapped out memory).
        - RAM VMS: Total virtual memory used by the process (includes swapped out memory).
        - System RAM used: Total RAM used by all processes.
        - System RAM total: Total available system RAM.
        - VRAM Allocated: GPU memory currently allocated by the process.
        - VRAM Reserved: GPU memory reserved by the process.
        - VRAM Peak Allocated: Peak GPU memory allocated by the process.
        - VRAM Peak Reserved: Peak GPU memory reserved by the process.
        """
        if not getattr(self, "profile_memory_enabled", True):
            return

        epoch = self.current_epoch + 1

        # System RAM metrics.
        if psutil is None:
            if not getattr(self, "_printed_psutil_warning", False):
                print("RAM usage: psutil is not installed; skipping RAM stats.")
                self._printed_psutil_warning = True
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            vm = psutil.virtual_memory()

            rss_gb = mem_info.rss / (1024 ** 3)  # Resident Set Size (physical memory used by process)
            vms_gb = mem_info.vms / (1024 ** 3)  # Virtual Memory Size (total virtual memory used by process)
            sys_used_gb = vm.used / (1024 ** 3)  # System RAM used by all processes
            sys_total_gb = vm.total / (1024 ** 3)  # Total system RAM

            if tb_logging:
                self.log("mem_ram_rss_gb", rss_gb, on_step=False, on_epoch=True, logger=True)
                self.log("mem_ram_vms_gb", vms_gb, on_step=False, on_epoch=True, logger=True)
                self.log("mem_ram_system_used_gb", sys_used_gb, on_step=False, on_epoch=True, logger=True)
                self.log("mem_ram_system_pct", float(vm.percent), on_step=False, on_epoch=True, logger=True)

            print(
                f"RAM after epoch {epoch}: "
                f"proc_rss={rss_gb:.2f} GB, proc_vms={vms_gb:.2f} GB, "
                f"system={sys_used_gb:.1f}/{sys_total_gb:.1f} GB ({vm.percent:.1f}%)"
            )

        # GPU VRAM metrics.
        if torch.cuda.is_available() and self.device.type == "cuda":
            device_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()

            vram_alloc_gb = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
            vram_reserved_gb = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
            vram_peak_alloc_gb = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 3)
            vram_peak_reserved_gb = torch.cuda.max_memory_reserved(device_idx) / (1024 ** 3)

            if tb_logging:
                self.log("mem_vram_alloc_gb", vram_alloc_gb, on_step=False, on_epoch=True, logger=True)
                self.log("mem_vram_reserved_gb", vram_reserved_gb, on_step=False, on_epoch=True, logger=True)
                self.log("mem_vram_peak_alloc_gb", vram_peak_alloc_gb, on_step=False, on_epoch=True, logger=True)
                self.log("mem_vram_peak_reserved_gb", vram_peak_reserved_gb, on_step=False, on_epoch=True, logger=True)

            print(
                f"VRAM after epoch {epoch}: "
                f"alloc={vram_alloc_gb:.2f} GB, reserved={vram_reserved_gb:.2f} GB, "
                f"peak_alloc={vram_peak_alloc_gb:.2f} GB, peak_reserved={vram_peak_reserved_gb:.2f} GB"
            )

            # Reset peak counters so next epoch reports epoch-local peak.
            torch.cuda.reset_peak_memory_stats(device_idx)
