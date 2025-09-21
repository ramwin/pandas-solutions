# SPDX-FileCopyrightText: 2025-present Xiang Wang <ramwin@qq.com>
#
# SPDX-License-Identifier: MIT

"""
DataFrameMatcher 的测试用例
"""

import unittest
import pandas as pd
import numpy as np
from pandas_solutions import DataFrameMatcher


class TestDataFrameMatcher(unittest.TestCase):
    """DataFrameMatcher 的测试类"""

    def setUp(self):
        """设置测试数据"""
        # 创建测试用的 DataFrame A
        self.df_a = pd.DataFrame({
            'address': ['addr1', 'addr1', 'addr1', 'addr2', 'addr2'],
            'data': ['data1_10', 'data1_20', 'data1_30', 'data2_15', 'data2_25'],
            'cycle': [10, 20, 30, 15, 25]
        })

        # 创建测试用的 DataFrame B
        self.df_b = pd.DataFrame({
            'cycle': [15, 25, 35, 12, 22],
            'address': ['addr1', 'addr1', 'addr1', 'addr2', 'addr2'],
            'way': [1, 2, 3, 1, 2],
            'set': [100, 200, 300, 400, 500]
        })

        self.matcher = DataFrameMatcher()

    def test_set_dataframes_valid(self):
        """测试设置有效的 DataFrame"""
        self.matcher.set_dataframes(self.df_a, self.df_b)
        stats = self.matcher.get_match_statistics()

        self.assertEqual(stats['df_a_rows'], 5)
        self.assertEqual(stats['df_b_rows'], 5)
        self.assertEqual(stats['unique_addresses_a'], 2)

    def test_set_dataframes_invalid_a(self):
        """测试设置无效的 DataFrame A（缺少必需字段）"""
        invalid_df_a = pd.DataFrame({
            'address': ['addr1'],
            'cycle': [10]
            # 缺少 'data' 字段
        })

        with self.assertRaises(ValueError) as context:
            self.matcher.set_dataframes(invalid_df_a, self.df_b)

        self.assertIn("DataFrame A 缺少必需字段", str(context.exception))

    def test_set_dataframes_invalid_b(self):
        """测试设置无效的 DataFrame B（缺少必需字段）"""
        invalid_df_b = pd.DataFrame({
            'cycle': [15],
            'address': ['addr1']
            # 缺少 'way' 和 'set' 字段
        })

        with self.assertRaises(ValueError) as context:
            self.matcher.set_dataframes(self.df_a, invalid_df_b)

        self.assertIn("DataFrame B 缺少必需字段", str(context.exception))

    def test_match_basic(self):
        """测试基本匹配功能"""
        self.matcher.set_dataframes(self.df_a, self.df_b)
        result = self.matcher.match()

        # 验证结果包含所有原始列加上 data 列
        expected_columns = {'cycle', 'address', 'way', 'set', 'data'}
        self.assertTrue(expected_columns.issubset(set(result.columns)))

        # 验证匹配结果
        # addr1, cycle=15 应该匹配到 addr1, cycle=10 的 data1_10
        addr1_cycle15 = result[(result['address'] == 'addr1') & (result['cycle'] == 15)]
        self.assertEqual(addr1_cycle15.iloc[0]['data'], 'data1_10')

        # addr1, cycle=25 应该匹配到 addr1, cycle=20 的 data1_20
        addr1_cycle25 = result[(result['address'] == 'addr1') & (result['cycle'] == 25)]
        self.assertEqual(addr1_cycle25.iloc[0]['data'], 'data1_20')

    def test_match_no_match(self):
        """测试无匹配情况"""
        # 创建一个 DataFrame B，其 cycle 值都小于 DataFrame A 中的最小值
        df_b_no_match = pd.DataFrame({
            'cycle': [5, 8],
            'address': ['addr1', 'addr2'],
            'way': [1, 2],
            'set': [100, 200]
        })

        self.matcher.set_dataframes(self.df_a, df_b_no_match)
        result = self.matcher.match()

        # 所有 data 列应该是 NaN
        self.assertTrue(result['data'].isna().all())

    def test_match_exact_match(self):
        """测试精确匹配"""
        # 创建 DataFrame B 中有与 DataFrame A 完全相同的 cycle 值
        df_b_exact = pd.DataFrame({
            'cycle': [10, 20],
            'address': ['addr1', 'addr1'],
            'way': [1, 2],
            'set': [100, 200]
        })

        self.matcher.set_dataframes(self.df_a, df_b_exact)
        result = self.matcher.match()

        # 验证精确匹配
        self.assertEqual(result.iloc[0]['data'], 'data1_10')
        self.assertEqual(result.iloc[1]['data'], 'data1_20')

    def test_match_without_dataframes(self):
        """测试未设置 DataFrame 时的匹配"""
        with self.assertRaises(ValueError) as context:
            self.matcher.match()

        self.assertIn("请先调用 set_dataframes 设置数据", str(context.exception))

    def test_get_statistics_empty(self):
        """测试空匹配器的统计信息"""
        stats = self.matcher.get_match_statistics()
        self.assertEqual(stats, {})

    def test_large_dataframe_performance(self):
        """测试大数据集性能"""
        # 创建较大的测试数据集
        np.random.seed(42)
        n_addresses = 1000
        n_records_a = 10000
        n_records_b = 50000

        addresses = [f'addr_{i}' for i in range(n_addresses)]

        # 创建 DataFrame A
        df_a_large = pd.DataFrame({
            'address': np.random.choice(addresses, n_records_a),
            'data': [f'data_{i}' for i in range(n_records_a)],
            'cycle': np.random.randint(1, 1000, n_records_a)
        })

        # 创建 DataFrame B
        df_b_large = pd.DataFrame({
            'cycle': np.random.randint(1, 1000, n_records_b),
            'address': np.random.choice(addresses, n_records_b),
            'way': np.random.randint(1, 10, n_records_b),
            'set': np.random.randint(100, 1000, n_records_b)
        })

        # 执行匹配
        matcher = DataFrameMatcher()
        matcher.set_dataframes(df_a_large, df_b_large)

        import time
        start_time = time.time()
        result = matcher.match()
        end_time = time.time()

        # 验证结果
        self.assertEqual(len(result), n_records_b)
        self.assertTrue('data' in result.columns)

        # 性能检查：应该在合理时间内完成（这里设置为10秒）
        execution_time = end_time - start_time
        self.assertLess(execution_time, 10.0, f"匹配耗时 {execution_time:.2f} 秒，超过预期")


if __name__ == '__main__':
    unittest.main()