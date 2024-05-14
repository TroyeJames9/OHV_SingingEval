# -*- coding: utf-8 -*-

"""

这个模块用来存放公用对象和函数

"""

import csv
import os
import chardet
import pandas as pd
import setting as se
from pathlib import Path


class CsvOp:
    """
    该类提供了一系列操作CSV文件的方法，包括读取、写入以及自动检测文件编码。其他功能有待开发
    作者：troyejames
    时间：20240514
    版本：v0.1
    """

    def __init__(self, file_path: Path = None):
        """
        初始化CsvOp实例，设置CSV文件路径和默认编码。

        参数:
        - file_path (Path, 可选): CSV文件的路径，默认为None。
        """
        self.file_path = file_path
        self.encoding = "utf-8"

    def _detect_encoding(self):
        """
        私有方法，用于自动检测CSV文件的编码方式。
        检测到的编码将更新实例的encoding属性。
        """
        with open(self.file_path, "rb") as file:
            result = chardet.detect(file.read())
        self.encoding = result["encoding"]

    def read(self):
        """
        读取CSV文件内容并返回为Pandas DataFrame。

        返回:
        - DataFrame: 包含CSV文件数据的Pandas DataFrame。
        """
        self._detect_encoding()
        df = pd.read_csv(self.file_path, encoding=self.encoding)
        return df

    def write(self, write_dict_list: list, overwrite: bool = False):
        """
        将数据字典列表写入或追加、覆写到CSV文件中。

        参数：
        - write_dict_list (list): 包含多个字典的列表，每个字典的键作为CSV列名，值作为列内容。
        - overwrite (bool, 默认为False): 如果为True，则覆写（覆盖）现有的CSV文件；否则在原有文件基础上追加数据。

        功能描述：
        此函数首先根据write_dict_list的第一个字典获取所有的字段名（即CSV列名），
        当指定的CSV文件不存在时，新建并写入表头和所有数据；
        若文件已存在且overwrite为True，则删除原有文件并重新写入所有数据；
        若文件已存在且overwrite为False，则在原有文件末尾追加数据。

        返回：
        无，但会在控制台输出CSV文件的创建或更新状态。
        """

        fieldnames = write_dict_list[0].keys()

        if os.path.exists(self.file_path):
            mode = "a"
            self._detect_encoding()
            if overwrite:
                os.remove(self.file_path)
                mode = "w"
        else:
            mode = "w"

        with open(
            self.file_path, mode=mode, newline="", encoding=self.encoding
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(fieldnames))

            # 对于新创建的文件，总是写入表头
            if mode == "w":
                writer.writeheader()

            # 写入或追加数据行
            writer.writerows(write_dict_list)

        print("CSV文件已创建或更新：", self.file_path)


def test_write_csv():
    j_score = CsvOp(file_path=se.RAW_DATA_DIR / "test.csv")
    write_dict_list = [{"what": 3}, {"what": 4}]
    j_score.write_csv(write_dict_list, overwrite=True)
    df = j_score.read()

    return df


if __name__ == "__main__":
    test_write_csv()
