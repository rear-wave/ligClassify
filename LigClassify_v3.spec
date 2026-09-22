# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path(SPECPATH)
model_root = project_root / "weights" / "multi_model"
data_files = [
    (str(model_root / "bundle.json"), "weights/multi_model"),
    (str(model_root / "type" / "model.pt"), "weights/multi_model/type"),
    (str(model_root / "NCG" / "model.pt"), "weights/multi_model/NCG"),
    (str(model_root / "NNBE" / "model.pt"), "weights/multi_model/NNBE"),
    (str(model_root / "PCG" / "model.pt"), "weights/multi_model/PCG"),
    (str(model_root / "PNBE" / "model.pt"), "weights/multi_model/PNBE"),
    (
        str(
            project_root
            / "configs"
            / "decision_nbe_human_202407_v3.json"
        ),
        "configs",
    ),
]

analysis = Analysis(
    [str(project_root / "portable_classify.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=data_files,
    hiddenimports=["scipy.signal"],
    hookspath=[str(project_root / "packaging_hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "PIL",
        "altair",
        "bitsandbytes",
        "datasets",
        "fsspec",
        "jinja2",
        "matplotlib",
        "pandas",
        "pyarrow",
        "pytest",
        "sklearn",
        "tensorflow",
        "torch.onnx",
        "torch.utils.tensorboard",
        "transformers",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)

cli_exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="LigClassify_v3_CPU",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

gui_exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="LigClassify_v3_CPU_GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    cli_exe,
    gui_exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="LigClassify_v3_CPU",
)
