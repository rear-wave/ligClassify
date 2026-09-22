"""Tkinter front end for the portable CPU v3 NBE classifier."""

from __future__ import annotations

import bisect
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import END, BooleanVar, StringVar, Text, Tk, filedialog, messagebox
from tkinter import ttk

from data.manifest import discover_date_inputs, discover_lig_files


_DATE_RE = re.compile(r"^\d{8}$")


class ClassifierGui:
    """Collect classification settings and supervise the CLI worker."""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.sources: list[str] = []
        self.stop_requested = False
        self.run_start = ""
        self.run_end = ""
        self.run_prefix = ""
        self.run_resume = False
        self.run_skip_io = False
        self.input_root = StringVar()
        self.start_date = StringVar()
        self.end_date = StringVar()
        self.output_dir = StringVar()
        self.prefix = StringVar(value="ZH_")
        self.batch_size = StringVar(value="64")
        self.resume = BooleanVar(value=False)
        self.skip_io = BooleanVar(value=False)
        self.status = StringVar(value="请选择输入和输出目录")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll)

    def _build(self) -> None:
        self.root.title("LigClassify v3 CPU 闪电分类")
        self.root.geometry("820x650")
        self.root.minsize(760, 580)
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="LigClassify v3 — CPU / NNBE + PNBE",
            font=("Microsoft YaHei UI", 15, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))
        frame.columnconfigure(1, weight=1)

        self._path_row(frame, 1, "输入根目录", self.input_root, self._pick_input)
        self._entry_row(frame, 2, "开始日期", self.start_date, "YYYYMMDD")
        self._entry_row(frame, 3, "结束日期", self.end_date, "YYYYMMDD")
        self._path_row(frame, 4, "输出目录", self.output_dir, self._pick_output)

        ttk.Label(frame, text="目录前缀").grid(row=5, column=0, sticky="w", pady=5)
        prefix = ttk.Combobox(
            frame,
            textvariable=self.prefix,
            values=("ZH_", "GZ_"),
            width=16,
        )
        prefix.grid(row=5, column=1, sticky="w", pady=5)
        ttk.Label(frame, text="例如 ZH_20230720").grid(
            row=5, column=2, sticky="w", padx=(8, 0)
        )
        self.inputs.append(prefix)

        self._entry_row(frame, 6, "批大小", self.batch_size, "CPU 建议 32–128")
        checks = ttk.Frame(frame)
        checks.grid(row=7, column=0, columnspan=3, sticky="w", pady=8)
        resume = ttk.Checkbutton(checks, text="断点续跑", variable=self.resume)
        skip = ttk.Checkbutton(
            checks,
            text="连续三次不可读时跳过文件",
            variable=self.skip_io,
        )
        resume.pack(side="left", padx=(0, 22))
        skip.pack(side="left")
        self.inputs.extend([resume, skip])

        buttons = ttk.Frame(frame)
        buttons.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(4, 10))
        self.start_button = ttk.Button(
            buttons, text="开始分类", command=self._start
        )
        self.stop_button = ttk.Button(
            buttons, text="安全停止", command=self._stop, state="disabled"
        )
        self.open_button = ttk.Button(
            buttons, text="打开输出目录", command=self._open_output
        )
        self.start_button.pack(side="left")
        self.stop_button.pack(side="left", padx=8)
        self.open_button.pack(side="left")

        self.progress = ttk.Progressbar(frame, maximum=100, mode="determinate")
        self.progress.grid(row=9, column=0, columnspan=3, sticky="ew")
        ttk.Label(frame, textvariable=self.status).grid(
            row=10, column=0, columnspan=3, sticky="w", pady=(5, 8)
        )
        self.log = Text(frame, height=16, wrap="word", state="disabled")
        self.log.grid(row=11, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(11, weight=1)
        scroll = ttk.Scrollbar(frame, command=self.log.yview)
        scroll.grid(row=11, column=3, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

    def _entry_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: StringVar,
        hint: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        ttk.Label(parent, text=hint).grid(
            row=row, column=2, sticky="w", padx=(8, 0)
        )
        self.inputs = getattr(self, "inputs", []) + [entry]

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: StringVar,
        command: object,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        button = ttk.Button(parent, text="浏览…", command=command)
        button.grid(row=row, column=2, sticky="ew", padx=(8, 0), pady=5)
        self.inputs = getattr(self, "inputs", []) + [entry, button]

    def _pick_input(self) -> None:
        selected = filedialog.askdirectory(title="选择包含日期目录的根目录")
        if selected:
            self.input_root.set(selected)

    def _pick_output(self) -> None:
        selected = filedialog.askdirectory(title="选择输出目录")
        if selected:
            self.output_dir.set(selected)

    def _validate(self) -> tuple[Path, Path, int]:
        source = Path(self.input_root.get().strip()).expanduser()
        destination = Path(self.output_dir.get().strip()).expanduser()
        if not source.is_dir():
            raise ValueError("输入根目录不存在")
        if not self.output_dir.get().strip():
            raise ValueError("请选择输出目录")
        if not _DATE_RE.fullmatch(self.start_date.get().strip()):
            raise ValueError("开始日期必须是 8 位 YYYYMMDD")
        if not _DATE_RE.fullmatch(self.end_date.get().strip()):
            raise ValueError("结束日期必须是 8 位 YYYYMMDD")
        if not self.prefix.get():
            raise ValueError("目录前缀不能为空")
        try:
            batch = int(self.batch_size.get())
        except ValueError as exc:
            raise ValueError("批大小必须是正整数") from exc
        if batch <= 0:
            raise ValueError("批大小必须是正整数")
        return source.resolve(), destination.resolve(), batch

    def _start(self) -> None:
        try:
            source, destination, batch = self._validate()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.stop_requested = False
        self.run_start = self.start_date.get().strip()
        self.run_end = self.end_date.get().strip()
        self.run_prefix = self.prefix.get()
        self.run_resume = bool(self.resume.get())
        self.run_skip_io = bool(self.skip_io.get())
        self.sources = []
        self.progress["value"] = 0
        self._set_running(True)
        self._append("正在扫描日期目录…\n")
        worker = threading.Thread(
            target=self._run,
            args=(source, destination, batch),
            daemon=True,
        )
        worker.start()

    def _command(self, source: Path, destination: Path, batch: int) -> list[str]:
        if getattr(sys, "frozen", False):
            executable = Path(sys.executable).with_name("LigClassify_v3_CPU.exe")
            command = [str(executable)]
        else:
            command = [sys.executable, str(Path(__file__).with_name("portable_classify.py"))]
        command.extend(
            [
                "--input_root", str(source),
                "--start_date", self.run_start,
                "--end_date", self.run_end,
                "--output_dir", str(destination),
                "--prefix", self.run_prefix,
                "--batch_size", str(batch),
            ]
        )
        if self.run_resume:
            command.append("--resume")
        if self.run_skip_io:
            command.append("--skip_io_errors")
        return command

    def _run(self, source: Path, destination: Path, batch: int) -> None:
        try:
            dates = discover_date_inputs(
                source,
                self.run_start,
                self.run_end,
                prefix=self.run_prefix,
            )
            files = discover_lig_files(
                source,
                search_roots=dates,
                skip_io_errors=self.run_skip_io,
            )
            self.sources = [relative for relative, _path in files]
            self.events.put(("count", len(files)))
            creationflags = 0
            startupinfo = None
            if os.name == "nt":
                creationflags = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.CREATE_NO_WINDOW
                )
            self.process = subprocess.Popen(
                self._command(source, destination, batch),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.events.put(("log", line))
            code = self.process.wait()
            self.events.put(("done", code))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.stop_requested = True
        self.status.set("正在安全停止并刷新已完成输出…")
        try:
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):
            messagebox.showwarning(
                "无法安全停止",
                "未能发送安全停止信号，请等待当前任务结束。",
            )

    def _poll_progress(self) -> None:
        if not self.sources:
            return
        state_path = Path(self.output_dir.get()) / ".classification_resume.json"
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            completed = str(payload["completed_source"])
        except (OSError, KeyError, json.JSONDecodeError):
            return
        count = bisect.bisect_right(self.sources, completed)
        percent = 100.0 * count / len(self.sources)
        self.progress["value"] = percent
        self.status.set(
            f"已完成 {count:,}/{len(self.sources):,} 个源文件（{percent:.1f}%）"
        )

    def _poll(self) -> None:
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append(str(value))
            elif kind == "count":
                self.status.set(f"发现 {int(value):,} 个 LIG 文件，正在分类…")
            elif kind == "done":
                self._finished(int(value))
            elif kind == "error":
                self._append(f"错误：{value}\n")
                self.status.set("任务启动失败")
                self._set_running(False)
                messagebox.showerror("分类失败", str(value))
        if self.process is not None and self.process.poll() is None:
            self._poll_progress()
        self.root.after(200, self._poll)

    def _finished(self, code: int) -> None:
        self.process = None
        self._set_running(False)
        if code == 0:
            self.progress["value"] = 100
            self.status.set("分类完成")
            messagebox.showinfo("完成", "分类任务已完成。")
        elif self.stop_requested:
            self.status.set("任务已停止，可勾选断点续跑后继续")
            messagebox.showinfo("已停止", "输出已刷新，可使用断点续跑继续。")
        else:
            self.status.set(f"分类异常结束（退出码 {code}）")
            messagebox.showerror("分类失败", "请查看下方运行日志。")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for widget in self.inputs:
            widget.configure(state=state)
        self.start_button.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(END, text)
        self.log.see(END)
        self.log.configure(state="disabled")

    def _open_output(self) -> None:
        path = Path(self.output_dir.get().strip())
        if path.is_dir():
            os.startfile(path)
        else:
            messagebox.showwarning("目录不存在", "输出目录尚未创建。")

    def _close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("任务运行中", "请先安全停止任务再关闭窗口。")
            return
        self.root.destroy()


def main() -> None:
    root = Tk()
    ClassifierGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
