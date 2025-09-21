# SPDX-FileCopyrightText: 2025-present Xiang Wang <ramwin@qq.com>
#
# SPDX-License-Identifier: MIT

"""
DataFrame 匹配器模块
用于高效匹配两个 DataFrame 之间的数据
"""

import logging
from typing import Optional, List, Dict, Any
import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)


class DataFrameMatcher:
    """
    DataFrame 匹配器类

    用于在两个 DataFrame 之间进行高效的数据匹配：
    - DataFrame A: 包含 address, data, cycle 字段
    - DataFrame B: 包含 cycle, address, way, set 字段

    目标：为 DataFrame B 的每一行找到 DataFrame A 中相同 address 下
    cycle 值小于等于当前行 cycle 的最大 cycle 记录，并添加对应的 data 信息
    """

    def __init__(self):
        """初始化匹配器"""
        self._df_a: Optional[pd.DataFrame] = None
        self._df_b: Optional[pd.DataFrame] = None
        self._indexed_df_a: Optional[pd.DataFrame] = None
        logger.info("DataFrameMatcher 初始化完成")

    def set_dataframes(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame
    ) -> None:
        """
        设置要匹配的两个 DataFrame

        Args:
            df_a: 包含 address, data, cycle 字段的 DataFrame
            df_b: 包含 cycle, address, way, set 字段的 DataFrame

        Raises:
            ValueError: 当 DataFrame 缺少必需字段时抛出异常
        """
        # 验证 DataFrame A 的字段
        required_cols_a = {'address', 'data', 'cycle'}
        if not required_cols_a.issubset(set(df_a.columns)):
            missing_cols = required_cols_a - set(df_a.columns)
            raise ValueError(f"DataFrame A 缺少必需字段: {missing_cols}")

        # 验证 DataFrame B 的字段
        required_cols_b = {'cycle', 'address', 'way', 'set'}
        if not required_cols_b.issubset(set(df_b.columns)):
            missing_cols = required_cols_b - set(df_b.columns)
            raise ValueError(f"DataFrame B 缺少必需字段: {missing_cols}")

        self._df_a = df_a.copy()
        self._df_b = df_b.copy()

        logger.info(f"设置 DataFrame A: {len(df_a)} 行, {len(df_a.columns)} 列")
        logger.info(f"设置 DataFrame B: {len(df_b)} 行, {len(df_b.columns)} 列")

        # 重置索引状态
        self._indexed_df_a = None

    def _build_index(self) -> None:
        """
        为 DataFrame A 建立索引以提高查询性能
        按 address 分组并对每组内的 cycle 进行排序
        """
        if self._df_a is None:
            raise ValueError("请先调用 set_dataframes 设置数据")

        logger.info("开始为 DataFrame A 建立索引...")

        # 按 address 和 cycle 排序，确保每个 address 组内 cycle 是有序的
        self._indexed_df_a = (self._df_a
                              .sort_values(['address', 'cycle'])
                              .reset_index(drop=True))

        unique_addresses = self._indexed_df_a['address'].nunique()
        logger.info(f"索引建立完成，共 {unique_addresses} 个唯一地址")

    def match(self) -> pd.DataFrame:
        """
        执行匹配操作

        Returns:
            pd.DataFrame: 包含匹配结果的 DataFrame B，添加了 data 列

        Raises:
            ValueError: 当未设置 DataFrame 时抛出异常
        """
        if self._df_a is None or self._df_b is None:
            raise ValueError("请先调用 set_dataframes 设置数据")

        logger.info("开始执行 DataFrame 匹配...")

        # 建立索引
        if self._indexed_df_a is None:
            self._build_index()

        # 由于 merge_asof 在某些情况下对排序要求严格，
        # 我们采用分组处理的方法确保正确性
        results = []
        original_index = self._df_b.reset_index(drop=False)

        for address in self._df_b['address'].unique():
            # 获取当前 address 的数据
            left_group = original_index[original_index['address'] == address].sort_values('cycle')
            right_group = self._indexed_df_a[
                self._indexed_df_a['address'] == address
            ][['cycle', 'data']].sort_values('cycle')

            if len(right_group) > 0:
                # 对每个 address 组进行 merge_asof
                merged = pd.merge_asof(
                    left=left_group,
                    right=right_group,
                    on='cycle',
                    direction='backward'
                )
                results.append(merged)
            else:
                # 如果 DataFrame A 中没有这个 address，添加 NaN data
                merged = left_group.copy()
                merged['data'] = None
                results.append(merged)

        # 合并所有结果并恢复原始顺序
        if results:
            result_df = pd.concat(results, ignore_index=True)
            result_df = result_df.set_index('index').sort_index()
            result_df.index.name = None
        else:
            # 如果没有任何匹配，返回原始 DataFrame B 加上空的 data 列
            result_df = self._df_b.copy()
            result_df['data'] = None

        matched_count = result_df['data'].notna().sum()
        total_count = len(result_df)
        match_rate = matched_count / total_count * 100

        logger.info(f"匹配完成: {matched_count}/{total_count} 行成功匹配 ({match_rate:.2f}%)")

        return result_df

    def get_match_statistics(self) -> Dict[str, Any]:
        """
        获取匹配统计信息

        Returns:
            Dict[str, Any]: 包含统计信息的字典
        """
        if self._df_a is None or self._df_b is None:
            return {}

        stats = {
            'df_a_rows': len(self._df_a),
            'df_b_rows': len(self._df_b),
            'unique_addresses_a': self._df_a['address'].nunique(),
            'unique_addresses_b': self._df_b['address'].nunique(),
            'df_a_cycle_range': (self._df_a['cycle'].min(), self._df_a['cycle'].max()),
            'df_b_cycle_range': (self._df_b['cycle'].min(), self._df_b['cycle'].max()),
        }

        logger.info(f"统计信息: {stats}")
        return stats