# -*- coding: utf-8 -*-

"""

这个模块用来存放公用对象和函数

"""

import csv
import os
import chardet
import pandas as pd
import setting
from pathlib import Path


class CsvOp:
    """
    pass
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.encoding = None
        self.header = None

    def _detect_encoding(self):
        with open(self.file_path, "rb") as file:
            result = chardet.detect(file.read())
        return result["encoding"]

    def read_csv(self):
        """
        pass
        """
        self.encoding = self._detect_encoding()
        df = pd.read_csv(self.file_path, encoding=self.encoding)

        return df


def test():
    j_score = CsvOp(file_path=setting.J_SCORE)
    judge_score = j_score.read_csv()

    return judge_score


if __name__ == "__main__":
    test()
