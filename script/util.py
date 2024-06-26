# -*- coding: utf-8 -*-

"""

这个模块用来存放公用对象和函数

"""

import csv
import os
import chardet
import torch
import torchaudio
import torchaudio.transforms as t
import pandas as pd
import setting as se
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
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


class AudioPreprocessor:
    """
    该类提供了一系列音频预处理方法，包括重采样、单声道化、波形图绘制、音频播放和频谱图绘制等。更多功能正在开发
    作者：troyejames
    时间：20240514
    版本：v0.1
    """

    def __init__(
        self,
        audio_path: str,
        resample_rate: int = 16000,
        resample_method: str = "kaiser_best",
    ):
        """
        初始化AudioPreprocessor实例，加载音频文件，并设置重采样率和重采样方法。

        参数:
        - audio_path (str): 音频文件的路径。
        - resample_rate (int, 可选): 目标采样率，默认为16000。
        - resample_method (str, 可选): 重采样方法，默认为"kaiser_best"。可选项包括
            hann_width64：使用hann窗，较高的低通滤波器宽度
            hann_width16：使用hann窗，较低的低通滤波器宽度
            kaiser_best：使用kaiser窗，同时其他参数为了得到高质量重采样而进行调整
            kaiser_fast：使用kaiser窗，同时其他参数为了提高重采样效率而进行调整
        """
        self.audio_path = audio_path
        self.waveform, self.sr = torchaudio.load(audio_path)
        self.resample_rate = resample_rate
        self.resample_method = resample_method

    def rspl(self):
        """
        根据指定的重采样方法对音频波形进行重采样，并对self.waveform和self.sr进行更新。

        返回:
        - waveform (Tensor): 重采样后的音频波形。
        """
        if self.resample_method == "hann_width64":
            resampler = t.Resample(
                self.sr,
                self.resample_rate,
                lowpass_filter_width=64,
                rolloff=0.99,
                resampling_method="sinc_interp_hann",
                dtype=self.waveform.dtype,
            )
            self.waveform, self.sr = resampler(self.waveform), self.resample_rate
            return self.waveform
        elif self.resample_method == "hann_width16":
            resampler = t.Resample(
                self.sr,
                self.resample_rate,
                lowpass_filter_width=16,
                rolloff=0.99,
                resampling_method="sinc_interp_hann",
                dtype=self.waveform.dtype,
            )
            self.waveform, self.sr = resampler(self.waveform), self.resample_rate
            return self.waveform
        elif self.resample_method == "kaiser_best":
            resampler = t.Resample(
                self.sr,
                self.resample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                dtype=self.waveform.dtype,
                beta=14.769656459379492,
            )
            self.waveform, self.sr = resampler(self.waveform), self.resample_rate
            return self.waveform
        elif self.resample_method == "kaiser_fast":
            resampler = t.Resample(
                self.sr,
                self.resample_rate,
                lowpass_filter_width=16,
                rolloff=0.85,
                resampling_method="sinc_interp_kaiser",
                dtype=self.waveform.dtype,
                beta=8.555504641634386,
            )
            self.waveform, self.sr = resampler(self.waveform), self.resample_rate
            return self.waveform
        else:
            raise ValueError(f"Unsupported resample method: {self.resample_method}")

    def mono(self):
        """
        将音频波形转换为单声道，并对self.waveform进行更新。

        返回:
        - waveform (Tensor): 单声道音频波形。
        """
        if self.waveform.ndim == 1:
            return self.waveform
        elif self.waveform.ndim == 3:  # 批处理情况
            self.waveform = torch.mean(self.waveform, dim=1)  # 沿着通道维度求平均
            return self.waveform
        elif self.waveform.ndim == 2:  # 单个样本情况
            # unsqueeze补充通道数信息
            self.waveform = torch.mean(self.waveform, dim=0).unsqueeze(0)
            return self.waveform
        else:
            raise ValueError("waveform should be a 2D or 3D tensor")

    def plot_waveform(self, title="Waveform", xlim=None, ylim=None):
        """
        绘制音频波形的波形图。

        参数:
        - title (str, 可选): 图表标题，默认为"Waveform"。
        - xlim (tuple, 可选): x轴的范围，默认为None。
        - ylim (tuple, 可选): y轴的范围，默认为None。
        """
        waveform = self.waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / self.sr

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)
        plt.show(block=False)

    def print_stats(self, src=None):
        """
        打印音频波形的统计信息。

        参数:
        - src (str, 可选): 音频源信息，默认为None。
        """
        if src:
            print("-" * 10)
            print("Source:", src)
            print("-" * 10)
        print("Sample Rate:", self.sr)
        print("Shape:", tuple(self.waveform.shape))
        print("Dtype:", self.waveform.dtype)
        print(f" - Max:     {self.waveform.max().item():6.3f}")
        print(f" - Min:     {self.waveform.min().item():6.3f}")
        print(f" - Mean:    {self.waveform.mean().item():6.3f}")
        print(f" - Std Dev: {self.waveform.std().item():6.3f}")
        print()
        print(self.waveform)
        print()

    def play_audio(self):
        """
        播放音频波形。
        """
        waveform = self.waveform.numpy()

        num_channels, num_frames = waveform.shape
        if num_channels == 1:
            display(Audio(waveform[0], rate=self.sr))
        elif num_channels == 2:
            display(Audio((waveform[0], waveform[1]), rate=self.sr))
        else:
            raise ValueError("Waveform with more than 2 channels are not supported.")

    def plot_specgram(self, title="Spectrogram", xlim=None):
        """
        绘制音频波形的频谱图。

        参数:
        - title (str, 可选): 图表标题，默认为"Spectrogram"。
        - xlim (tuple, 可选): x轴的范围，默认为None。
        """
        waveform = self.waveform.numpy()

        num_channels, num_frames = waveform.shape

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=self.sr)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show(block=False)


class JudgeCorrelationCalculator(CsvOp):
    """
    该类用于计算评委之间的皮尔逊相关系数。
    它继承自CsvOp类，并添加了用于计算相关系数的特有方法。
    作者：troyejames
    时间：20240603
    版本：v0.1
    """

    def __init__(
        self,
        file_path: Path,
        judge_col: str,
        module_col: str,
        sort_col: str,
        score_col: str,
    ):
        """
        初始化JudgeCorrelationCalculator实例，设置CSV文件路径和相关的列名。

        参数:
        - file_path (Path): CSV文件的路径。这里特指含有各评委评分数据的csv文件。
        - judge_col (str): 对评委具有唯一标识的列的名称。
        - module_col (str): 对模块具有唯一标识的列的名称。
        - sort_col (str): 用于排序的列的名称。通常为数据表的主键id
        - score_col (str): 评委评分数值结果的列的名称。
        """
        super().__init__(file_path)  # 调用基类的__init__方法
        self.judge_col = judge_col
        self.module_col = module_col
        self.sort_col = sort_col
        self.score_col = score_col

    @staticmethod
    def _cal_avg_pearson(arrays: dict):
        """
        私有静态方法，用于计算平均皮尔逊相关系数。
        它接收一个字典，其中键是评分模块名称+评委id，值是评委的评分数组。

        参数:
        - arrays (dict): 一个字典，其中键是模块名称+评委id，值是评委的评分ndarray数组。

        返回:
        - float: 所有评委之间评分的平均皮尔逊相关系数。
        """
        # 将所有数组组合成一个大的二维数组
        combined_array = np.array([arrays[key] for key in arrays.keys()])
        # 计算各列之间的皮尔逊相关系数
        correlation_matrix = np.corrcoef(combined_array)
        row_indices, col_indices = np.triu_indices(correlation_matrix.shape[0], k=1)
        average_lower_triangle = np.mean(correlation_matrix[row_indices, col_indices])

        return average_lower_triangle

    def get_per_judge_module(self):
        """
        提取每个评委在每个模块的评分数据，分别存储为字典中的一个ndarray元素

        返回:
        - dict: 一个字典，其中键是模块名称+评委id，值是评委的评分数组。
        """
        judge_pd = self.read()
        per_judge_by_module_dict = {}
        judge_id_list = judge_pd[self.judge_col].unique()
        module_col_list = judge_pd[self.module_col].unique()
        for judge_id in judge_id_list:
            for module in module_col_list:
                per_judge_by_module = judge_pd[
                    (judge_pd[self.judge_col] == judge_id)
                    & (judge_pd[self.module_col] == module)
                ].sort_values(by=self.sort_col)
                key_name = str(judge_id) + "_" + module
                per_judge_by_module_dict[key_name] = per_judge_by_module[
                    self.score_col
                ].to_numpy()

        return per_judge_by_module_dict

    def cal_per_module_pearson(self, arrays: dict):
        """
        计算每个模块的平均皮尔逊相关系数。

        参数:
        - arrays (dict): 一个字典，其中键是模块名称+评委id，值是评委的评分数组。

        返回:
        - dict: 一个字典，其中键是模块名称，值是该模块的评委间评分的平均皮尔逊相关系数。
        """
        unique_module = {key.split("_")[1] for key in arrays.keys() if "_" in key}
        pearson_dict = {}
        for module in unique_module:
            module_dict = {key: value for key, value in arrays.items() if module in key}
            pearson_dict[module] = self._cal_avg_pearson(module_dict)

        return pearson_dict


def test_write():
    j_score = CsvOp(file_path=se.RAW_DATA_DIR / "test.csv")
    write_dict_list = [{"what": 3}, {"what": 4}]
    j_score.write(write_dict_list, overwrite=True)
    df = j_score.read()

    return df


def test_audio_pre():
    cst_wav = se.UPLOAD_FILE_DIR / "qilai" / "cst.wav"
    ap = AudioPreprocessor(str(cst_wav.resolve()))
    ap.rspl()
    ap.mono()
    return ap


def test_judge_corr():
    jtest = JudgeCorrelationCalculator(
        file_path=se.J_SCORE,
        judge_col="judge_id",
        module_col="score_dim",
        sort_col="person",
        score_col="judge_score",
    )
    rs_dict = jtest.get_per_judge_module()
    p_rs = jtest.cal_per_module_pearson(rs_dict)
    return p_rs


if __name__ == "__main__":
    print(test_judge_corr())
