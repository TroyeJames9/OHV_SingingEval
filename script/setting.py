# -*- coding: utf-8 -*-

"""所有跨文件的全局变量由本模块计算并赋值

其他模块需要使用本模块全局变量时，在模块开头导入本模块即可
例子：
    from setting import *...

"""

import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # program root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# data
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"

J_SCORE = RAW_DATA_DIR / "judges_score.csv"

# audio
UPLOAD_FILE_DIR = ROOT / "audio"
