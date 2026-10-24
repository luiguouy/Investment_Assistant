# -*- coding: utf-8 -*-
import os
import sys

# 确保在任何工作目录下执行测试时，项目根目录都在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
