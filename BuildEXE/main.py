#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELT 生态廊道分析工具 — 主程序（GUI 入口）
==========================================
运行方式:
    python main.py
    或打包为 ELT_Tool.exe 后双击运行

界面功能:
    - 选择 3 个输入文件（生态源地、连通性、阻力面）
    - 选择输出目录
    - 支持逐步运行（步骤 0~3）或一键运行全部
    - 日志窗口实时显示进度
"""

import sys
import os

# ── 解决 PyInstaller --windowed 模式下 sys.stdout/stderr 为 None 的问题 ──────
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from typing import Optional, Union, Dict, Any

# ── PyInstaller 打包后设置 GDAL/PROJ 数据路径 ────────────
if getattr(sys, 'frozen', False):
    _bundle = sys._MEIPASS
    # 增加对 PROJ_LIB 的设置，部分旧版 GDAL/PROJ 仍使用它
    os.environ['PROJ_LIB'] = os.path.join(_bundle, 'pyproj', 'proj_data')
    os.environ['PROJ_DATA'] = os.path.join(_bundle, 'pyproj', 'proj_data')
    os.environ['GDAL_DATA'] = os.path.join(_bundle, 'rasterio', 'gdal_data')
    
    # 尝试在打包环境中添加 DLL 目录（针对 Python 3.8+）
    if hasattr(os, 'add_dll_directory'):
        # 捆绑后的 DLLs 通常直接在根目录或特定子目录下
        os.add_dll_directory(_bundle)
        # 针对 pip 安装的 wheel 版本，可能存在 .libs 目录
        for lib_dir in ['rasterio.libs', 'fiona.libs', 'pyproj.libs', 'shapely.libs']:
            full_lib_path = os.path.join(_bundle, lib_dir)
            if os.path.isdir(full_lib_path):
                os.add_dll_directory(full_lib_path)

# ── 确保当前目录在 sys.path 中（开发模式）─────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import write_data_description
from step0 import run_preprocess
from step1 import run_adjacent
from step2 import run_activate
from step3 import run_corridor


# ════════════════════════════════════════════════
# GUI 主类
# ════════════════════════════════════════════════

class ELTApp:
    """生态廊道分析工具主界面。"""

    _WIN_W = 820
    _WIN_H = 680

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ELT 生态链接工具 v1.0")
        self.root.geometry(f"{self._WIN_W}x{self._WIN_H}")
        self.root.resizable(True, True)

        self._vars: dict[str, tk.StringVar] = {}
        self._running = False

        self._build_ui()
        self._set_defaults()

    # ────────────────────────────────────────────
    # UI 构建
    # ────────────────────────────────────────────

    def _build_ui(self):
        """搭建界面布局。"""
        pad = {"padx": 10, "pady": 4}

        # ── 输入文件区 ──────────────────────────
        f_input = ttk.LabelFrame(self.root, text=" 输入文件 ", padding=8)
        f_input.pack(fill='x', **pad)

        input_fields = [
            ("eco_source",   "生态源地栅格 (源地值为1，非源地值为0)", "tif"),
            ("resistance",   "阻力面栅格 (确保与源地栅格分辨率范围一致)", "tif"),
            ("connectivity", "属性栅格 (建议使用连通性，或用阻力面栅格)", "tif"),
        ]
        for key, label, ext in input_fields:
            self._add_file_row(f_input, key, label, ext)

        row_doc = ttk.Frame(f_input)
        row_doc.pack(fill='x', pady=2)
        ttk.Button(row_doc, text="软件文档", width=10, command=self._open_docs).pack(side='right')

        # ── 输出目录区 ──────────────────────────
        f_output = ttk.LabelFrame(self.root, text=" 输出目录 ", padding=8)
        f_output.pack(fill='x', **pad)
        self._add_dir_row(f_output, "output_dir", "输出目录")

        # ── 参数区 ──────────────────────────────
        f_param = ttk.LabelFrame(self.root, text=" 参数设置 ", padding=8)
        f_param.pack(fill='x', **pad)

        row = ttk.Frame(f_param)
        row.pack(fill='x')
        ttk.Label(row, text="最小斑块面积 (km²) :").pack(side='left')
        self._vars['min_area'] = tk.StringVar(value="1")
        ttk.Entry(row, textvariable=self._vars['min_area'], width=8).pack(side='left', padx=6)
        ttk.Label(row, text="(预处理部分源地的面积过滤阈值)", foreground='gray').pack(side='left')

        # ── 按钮区 ──────────────────────────────
        f_btn = ttk.Frame(self.root)
        f_btn.pack(fill='x', **pad)

        ttk.Button(f_btn, text="▶ 运行全部步骤",
                   command=self._run_all, width=16).pack(side='left', padx=4)

        for step_i, label in enumerate(
            ["步骤 0: 预处理", "步骤 1: 邻接", "步骤 2: 激活点", "步骤 3: 廊道"]
        ):
            ttk.Button(f_btn, text=label,
                       command=lambda s=step_i: self._run_step(s),
                       width=14).pack(side='left', padx=2)

        ttk.Button(f_btn, text="清空日志",
                   command=self._clear_log, width=10).pack(side='right', padx=4)

        # ── 进度条 ──────────────────────────────
        self._progress = ttk.Progressbar(self.root, mode='indeterminate')
        self._progress.pack(fill='x', padx=10, pady=2)

        # ── 日志区 ──────────────────────────────
        f_log = ttk.LabelFrame(self.root, text=" 运行日志 ", padding=5)
        f_log.pack(fill='both', expand=True, **pad)

        self._log_text = scrolledtext.ScrolledText(
            f_log, height=16, state='disabled',
            font=('Consolas', 9), bg='#1e1e1e', fg='#d4d4d4',
            insertbackground='white'
        )
        self._log_text.pack(fill='both', expand=True)

    def _add_file_row(self, parent, key, label, ext):
        """添加一行文件选择控件。"""
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=2)

        ttk.Label(row, text=label, width=38, anchor='w').pack(side='left')
        var = tk.StringVar()
        self._vars[key] = var
        ttk.Entry(row, textvariable=var, width=36).pack(side='left', padx=4)
        ttk.Button(
            row, text="浏览…", width=7,
            command=lambda v=var, e=ext: self._browse_file(v, e)
        ).pack(side='left')

    def _add_dir_row(self, parent, key, label):
        """添加一行目录选择控件。"""
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=2)

        ttk.Label(row, text=label, width=38, anchor='w').pack(side='left')
        var = tk.StringVar()
        self._vars[key] = var
        ttk.Entry(row, textvariable=var, width=36).pack(side='left', padx=4)
        ttk.Button(row, text="浏览…", width=7,
                   command=lambda v=var: self._browse_dir(v)).pack(side='left')

    # ────────────────────────────────────────────
    # 默认路径推断
    # ────────────────────────────────────────────

    def _set_defaults(self):
        """若与本脚本同级存在 InputData/，则自动填充默认路径。"""
        here      = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(os.path.dirname(here), 'InputData')
        out_dir   = os.path.join(os.path.dirname(here), 'OutputData')

        defaults = {
            'eco_source'  : os.path.join(input_dir, 'EcologicalSource.tif'),
            'connectivity': os.path.join(input_dir, 'Connectivity.tif'),
            'resistance'  : os.path.join(input_dir, 'ResistanceSurface.tif'),
            'output_dir'  : out_dir,
        }
        for key, path in defaults.items():
            if os.path.exists(path) or key == 'output_dir':
                self._vars[key].set(path)

    # ────────────────────────────────────────────
    # 文件/目录选择
    # ────────────────────────────────────────────

    def _browse_file(self, var, ext):
        path = filedialog.askopenfilename(
            filetypes=[(f"栅格文件 (*.{ext})", f"*.{ext}"), ("所有文件", "*.*")]
        )
        if path:
            var.set(path)

    def _browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _open_docs(self):
        url = "https://github.com/HaoweiGis/Ecological-Linkage-Tool"
        try:
            webbrowser.open_new_tab(url)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开链接:\n{url}\n\n{type(e).__name__}: {e}")

    # ────────────────────────────────────────────
    # 日志
    # ────────────────────────────────────────────

    def _log(self, msg: str):
        """线程安全地向日志窗口追加消息。"""
        self._log_text.config(state='normal')
        self._log_text.insert('end', msg + '\n')
        self._log_text.see('end')
        self._log_text.config(state='disabled')
        self.root.update_idletasks()

    def _clear_log(self):
        self._log_text.config(state='normal')
        self._log_text.delete('1.0', 'end')
        self._log_text.config(state='disabled')

    # ────────────────────────────────────────────
    # 路径收集
    # ────────────────────────────────────────────

    def _collect_paths(self) -> Optional[dict]:
        """
        收集并验证所有输入路径。
        返回 None 表示验证失败。
        """
        eco     = self._vars['eco_source'].get().strip()
        conn    = self._vars['connectivity'].get().strip()
        resis   = self._vars['resistance'].get().strip()
        out_dir = self._vars['output_dir'].get().strip()

        if not eco or not os.path.isfile(eco):
            messagebox.showerror("错误", f"生态源地栅格文件不存在:\n{eco}")
            return None

        if not out_dir:
            out_dir = os.path.join(os.path.dirname(eco), 'OutputData')

        os.makedirs(out_dir, exist_ok=True)

        try:
            min_area = float(self._vars['min_area'].get())
        except ValueError:
            min_area = 1.0

        return {
            'eco_source'  : eco,
            'connectivity': conn,
            'resistance'  : resis,
            'output_dir'  : out_dir,
            'min_area'    : min_area,
        }

    # ────────────────────────────────────────────
    # 运行控制
    # ────────────────────────────────────────────

    def _run_all(self):
        if self._running:
            return
        threading.Thread(target=self._thread_run_all, daemon=True).start()

    def _run_step(self, step: int):
        if self._running:
            return
        threading.Thread(target=self._thread_run_step, args=(step,), daemon=True).start()

    def _thread_run_all(self):
        paths = self._collect_paths()
        if paths is None:
            return
        write_data_description(paths['output_dir'])
        self._start_progress()
        try:
            for step in range(4):
                self._execute_step(step, paths)
        except Exception as e:
            self._log(f"\n[错误] {type(e).__name__}: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            self._stop_progress()

    def _thread_run_step(self, step: int):
        paths = self._collect_paths()
        if paths is None:
            return
        write_data_description(paths['output_dir'])
        self._start_progress()
        try:
            self._execute_step(step, paths)
        except Exception as e:
            self._log(f"\n[错误] {type(e).__name__}: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            self._stop_progress()

    def _start_progress(self):
        self._running = True
        self._progress.start(12)

    def _stop_progress(self):
        self._progress.stop()
        self._running = False

    def _execute_step(self, step: int, paths: dict):
        """分发到各步骤函数。"""
        out = paths['output_dir']

        if step == 0:
            run_preprocess(
                paths['eco_source'], out,
                min_area_km2=paths['min_area'],
                log_callback=self._log,
            )

        elif step == 1:
            filled_tif = os.path.join(out, 'mspa_core_filled.tif')
            if not os.path.isfile(filled_tif):
                raise FileNotFoundError(f"找不到步骤 0 输出: {filled_tif}\n请先运行步骤 0。")
            run_adjacent(filled_tif, out, log_callback=self._log)

        elif step == 2:
            filled_tif = os.path.join(out, 'mspa_core_filled.tif')
            adj_csv    = os.path.join(out, 'adjacency.csv')
            for p in [filled_tif, adj_csv]:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"找不到文件: {p}\n请先运行前置步骤。")
            run_activate(filled_tif, adj_csv, out, log_callback=self._log)

        elif step == 3:
            core_tif   = os.path.join(out, 'mspa_core_filled.tif')
            adj_csv    = os.path.join(out, 'adjacency.csv')
            conn_tif   = paths['connectivity']
            resis_tif  = paths['resistance']

            for label, p in [
                ("源地栅格",    core_tif),
                ("邻接关系表",  adj_csv),
                ("连通性栅格",  conn_tif),
                ("阻力面栅格",  resis_tif),
            ]:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"找不到 {label}: {p}")

            run_corridor(
                None, core_tif, conn_tif, resis_tif,
                adj_csv, out, log_callback=self._log,
            )


# ════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════

def main():
    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.25)   # 高 DPI 适配
    except Exception:
        pass
    ELTApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
