#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import tempfile
import unittest
from pathlib import Path
from typing import List

import pandas as pd

from pandas_solutions.merger import Merger


class TestMerger(unittest.TestCase):
    """Merger类的测试用例"""

    def setUp(self) -> None:
        """测试准备"""
        self.merger = Merger(max_workers=2)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        """测试清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _create_test_csv_files(self, file_configs: List[dict]) -> None:
        """创建测试用的CSV文件

        Args:
            file_configs: 文件配置列表，每个配置包含文件名和数据
        """
        for config in file_configs:
            file_path = self.temp_dir / config['filename']
            df = pd.DataFrame(config['data'])

            if file_path.suffix.lower() == '.gz':
                df.to_csv(file_path, index=False, compression='gzip')
            else:
                df.to_csv(file_path, index=False)

    def test_init_merger(self) -> None:
        """测试Merger初始化"""
        # 测试默认初始化
        merger_default = Merger()
        self.assertGreater(merger_default.max_workers, 0)

        # 测试指定工作进程数
        merger_custom = Merger(max_workers=4)
        self.assertEqual(merger_custom.max_workers, 4)

    def test_find_csv_files(self) -> None:
        """测试查找CSV文件功能"""
        # 创建测试文件
        files_config = [
            {'filename': 'test1.csv', 'data': {'col1': [1, 2], 'col2': ['a', 'b']}},
            {'filename': 'test2.csv.gz', 'data': {'col1': [3, 4], 'col2': ['c', 'd']}},
            {'filename': 'not_csv.txt', 'data': {'col1': [5, 6], 'col2': ['e', 'f']}},
        ]

        # 创建CSV文件
        for config in files_config[:2]:  # 只创建CSV文件
            self._create_test_csv_files([config])

        # 创建非CSV文件
        (self.temp_dir / 'not_csv.txt').write_text('some text')

        csv_files = self.merger._find_csv_files(self.temp_dir)

        # 验证找到了正确的文件
        self.assertEqual(len(csv_files), 2)
        file_names = [f.name for f in csv_files]
        self.assertIn('test1.csv', file_names)
        self.assertIn('test2.csv.gz', file_names)
        self.assertNotIn('not_csv.txt', file_names)

    def test_merge_basic_csv_files(self) -> None:
        """测试基本的CSV文件合并功能"""
        files_config = [
            {
                'filename': 'file1.csv',
                'data': {'id': [1, 2], 'name': ['Alice', 'Bob'], 'score': [90, 85]}
            },
            {
                'filename': 'file2.csv',
                'data': {'id': [3, 4], 'name': ['Charlie', 'David'], 'score': [88, 92]}
            }
        ]

        self._create_test_csv_files(files_config)
        output_file = self.output_dir / 'merged.csv'

        # 执行合并
        self.merger.merge_csv_files(
            input_dir=self.temp_dir,
            output_file=output_file
        )

        # 验证结果
        self.assertTrue(output_file.exists())
        result_df = pd.read_csv(output_file)

        # 验证数据完整性
        self.assertEqual(len(result_df), 4)
        self.assertListEqual(list(result_df.columns), ['id', 'name', 'score'])
        self.assertListEqual(list(result_df['id']), [1, 2, 3, 4])

    def test_merge_compressed_files(self) -> None:
        """测试压缩文件的合并"""
        files_config = [
            {
                'filename': 'file1.csv.gz',
                'data': {'value': [1, 2, 3], 'category': ['A', 'B', 'C']}
            },
            {
                'filename': 'file2.csv.gz',
                'data': {'value': [4, 5, 6], 'category': ['D', 'E', 'F']}
            }
        ]

        self._create_test_csv_files(files_config)
        output_file = self.output_dir / 'merged_compressed.csv.gz'

        # 执行合并
        self.merger.merge_csv_files(
            input_dir=self.temp_dir,
            output_file=output_file,
            compression_level=6
        )

        # 验证结果
        self.assertTrue(output_file.exists())
        result_df = pd.read_csv(output_file, compression='gzip')

        self.assertEqual(len(result_df), 6)
        self.assertListEqual(list(result_df['value']), [1, 2, 3, 4, 5, 6])

    def test_merge_with_sorting(self) -> None:
        """测试带排序的合并功能"""
        files_config = [
            {
                'filename': 'file1.csv',
                'data': {'timestamp': [3, 1], 'data': ['third', 'first']}
            },
            {
                'filename': 'file2.csv',
                'data': {'timestamp': [4, 2], 'data': ['fourth', 'second']}
            }
        ]

        self._create_test_csv_files(files_config)
        output_file = self.output_dir / 'sorted_merged.csv'

        # 执行带排序的合并
        self.merger.merge_csv_files(
            input_dir=self.temp_dir,
            output_file=output_file,
            sort_fields=['timestamp']
        )

        # 验证结果
        result_df = pd.read_csv(output_file)
        self.assertListEqual(
            list(result_df['data']),
            ['first', 'second', 'third', 'fourth']
        )
        self.assertListEqual(
            list(result_df['timestamp']),
            [1, 2, 3, 4]
        )

    def test_merge_multi_column_sorting(self) -> None:
        """测试多列排序"""
        files_config = [
            {
                'filename': 'file1.csv',
                'data': {
                    'group': ['A', 'B', 'A'],
                    'priority': [2, 1, 1],
                    'value': [10, 20, 30]
                }
            }
        ]

        self._create_test_csv_files(files_config)
        output_file = self.output_dir / 'multi_sorted.csv'

        # 按group, priority排序
        self.merger.merge_csv_files(
            input_dir=self.temp_dir,
            output_file=output_file,
            sort_fields=['group', 'priority']
        )

        result_df = pd.read_csv(output_file)

        # 验证排序结果
        expected_groups = ['A', 'A', 'B']
        expected_priorities = [1, 2, 1]
        expected_values = [30, 10, 20]

        self.assertListEqual(list(result_df['group']), expected_groups)
        self.assertListEqual(list(result_df['priority']), expected_priorities)
        self.assertListEqual(list(result_df['value']), expected_values)

    def test_empty_directory(self) -> None:
        """测试空目录的处理"""
        empty_dir = self.temp_dir / 'empty'
        empty_dir.mkdir()
        output_file = self.output_dir / 'empty_merge.csv'

        with self.assertRaises(ValueError) as context:
            self.merger.merge_csv_files(
                input_dir=empty_dir,
                output_file=output_file
            )

        self.assertIn("没有找到CSV文件", str(context.exception))

    def test_nonexistent_directory(self) -> None:
        """测试不存在目录的处理"""
        nonexistent_dir = self.temp_dir / 'nonexistent'
        output_file = self.output_dir / 'nonexistent_merge.csv'

        with self.assertRaises(ValueError) as context:
            self.merger.merge_csv_files(
                input_dir=nonexistent_dir,
                output_file=output_file
            )

        self.assertIn("输入目录不存在", str(context.exception))

    def test_compression_levels(self) -> None:
        """测试不同压缩等级"""
        files_config = [
            {
                'filename': 'test.csv',
                'data': {'data': list(range(100))}  # 更多数据以便测试压缩
            }
        ]

        self._create_test_csv_files(files_config)

        # 测试不同压缩等级
        for level in [1, 6, 9]:
            output_file = self.output_dir / f'compressed_level_{level}.csv.gz'

            self.merger.merge_csv_files(
                input_dir=self.temp_dir,
                output_file=output_file,
                compression_level=level
            )

            # 验证文件存在且可读
            self.assertTrue(output_file.exists())
            result_df = pd.read_csv(output_file, compression='gzip')
            self.assertEqual(len(result_df), 100)

    def test_mixed_file_types(self) -> None:
        """测试混合文件类型（.csv 和 .csv.gz）"""
        files_config = [
            {
                'filename': 'regular.csv',
                'data': {'type': ['regular'], 'value': [1]}
            },
            {
                'filename': 'compressed.csv.gz',
                'data': {'type': ['compressed'], 'value': [2]}
            }
        ]

        self._create_test_csv_files(files_config)
        output_file = self.output_dir / 'mixed_merge.csv'

        self.merger.merge_csv_files(
            input_dir=self.temp_dir,
            output_file=output_file
        )

        result_df = pd.read_csv(output_file)
        self.assertEqual(len(result_df), 2)
        self.assertSetEqual(set(result_df['type']), {'regular', 'compressed'})

    def test_large_number_of_files(self) -> None:
        """测试处理大量文件"""
        # 创建多个小文件
        files_config = []
        for i in range(10):
            files_config.append({
                'filename': f'file_{i:02d}.csv',
                'data': {'file_id': [i], 'data': [f'data_{i}']}
            })

        self._create_test_csv_files(files_config)
        output_file = self.output_dir / 'many_files_merge.csv'

        self.merger.merge_csv_files(
            input_dir=self.temp_dir,
            output_file=output_file,
            sort_fields=['file_id']
        )

        result_df = pd.read_csv(output_file)
        self.assertEqual(len(result_df), 10)
        self.assertListEqual(list(result_df['file_id']), list(range(10)))


if __name__ == '__main__':
    unittest.main()