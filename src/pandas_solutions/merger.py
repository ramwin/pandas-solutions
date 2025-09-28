#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Xiang Wang <ramwin@qq.com>


from .types import Dir, File


class Merger:
    """
    合并一个文件夹的多个csv
    1. 注意,要用多进程
    """
