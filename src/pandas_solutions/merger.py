#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Xiang Wang <ramwin@qq.com>

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .types import Dir, File

logger = logging.getLogger(__name__)


def _read_csv_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """读取单个CSV文件的辅助函数，用于多进程处理

    Args:
        file_path: CSV文件路径

    Returns:
        读取的DataFrame

    Raises:
        Exception: 当文件读取失败时
    """
    try:
        file_path = Path(file_path)
        logger.debug(f"正在读取文件: {file_path}")

        if file_path.suffix.lower() == '.gz':
            df = pd.read_csv(file_path, compression='gzip')
        else:
            df = pd.read_csv(file_path)

        logger.debug(f"成功读取文件 {file_path}，数据形状: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"读取文件 {file_path} 失败: {e}")
        raise


class Merger:
    """
    合并一个文件夹的多个csv文件

    支持功能：
    1. 多进程并行读取文件
    2. 支持排序字段
    3. 压缩保存
    """

    def __init__(self, max_workers: Optional[int] = None):
        """初始化Merger

        Args:
            max_workers: 最大工作进程数，默认为CPU核心数
        """
        self.max_workers = max_workers or mp.cpu_count()
        logger.info(f"Merger初始化完成，最大工作进程数: {self.max_workers}")

    def merge_csv_files(
        self,
        input_dir: Union[str, Path, Dir],
        output_file: Union[str, Path, File],
        sort_fields: Optional[List[str]] = None,
        compression_level: int = 6
    ) -> None:
        """合并指定文件夹内的所有CSV文件

        Args:
            input_dir: 输入文件夹路径
            output_file: 输出文件路径
            sort_fields: 排序字段列表，为None时不排序
            compression_level: 压缩等级 (0-9)，默认为6

        Raises:
            ValueError: 当输入目录不存在或没有找到CSV文件时
            Exception: 其他处理错误
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)

        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"输入目录不存在或不是目录: {input_dir}")

        logger.info(f"开始合并目录 {input_dir} 中的CSV文件")

        # 查找所有CSV文件
        csv_files = self._find_csv_files(input_dir)
        if not csv_files:
            raise ValueError(f"在目录 {input_dir} 中没有找到CSV文件")

        logger.info(f"找到 {len(csv_files)} 个CSV文件")

        # 使用多进程读取文件
        dataframes = self._read_files_parallel(csv_files)

        # 合并DataFrame
        logger.info("开始合并DataFrame")
        merged_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"合并完成，最终数据形状: {merged_df.shape}")

        # 排序（如果指定了排序字段）
        if sort_fields:
            logger.info(f"按字段 {sort_fields} 进行排序")
            merged_df = merged_df.sort_values(by=sort_fields)
            merged_df = merged_df.reset_index(drop=True)

        # 保存文件
        self._save_dataframe(merged_df, output_file, compression_level)
        logger.info(f"文件合并完成，保存到: {output_file}")

    def _find_csv_files(self, input_dir: Path) -> List[Path]:
        """查找目录中的所有CSV文件

        Args:
            input_dir: 输入目录

        Returns:
            CSV文件路径列表
        """
        csv_files: List[Path] = []

        # 查找 .csv 文件
        csv_files.extend(input_dir.glob("*.csv"))

        # 查找 .csv.gz 文件
        csv_files.extend(input_dir.glob("*.csv.gz"))

        # 按文件名排序，确保处理顺序一致
        csv_files.sort()

        logger.debug(f"找到的CSV文件: {[f.name for f in csv_files]}")
        return csv_files

    def _read_files_parallel(self, csv_files: List[Path]) -> List[pd.DataFrame]:
        """使用多进程并行读取CSV文件

        Args:
            csv_files: CSV文件路径列表

        Returns:
            DataFrame列表
        """
        logger.info(f"使用 {self.max_workers} 个进程并行读取文件")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            dataframes = list(executor.map(_read_csv_file, csv_files))

        return dataframes

    def _save_dataframe(
        self,
        df: pd.DataFrame,
        output_file: Path,
        compression_level: int
    ) -> None:
        """保存DataFrame到文件

        Args:
            df: 要保存的DataFrame
            output_file: 输出文件路径
            compression_level: 压缩等级
        """
        logger.info(f"保存DataFrame到 {output_file}，压缩等级: {compression_level}")

        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.suffix.lower() == '.gz':
            df.to_csv(
                output_file,
                index=False,
                compression={'method': 'gzip', 'compresslevel': compression_level}
            )
        else:
            df.to_csv(output_file, index=False)
