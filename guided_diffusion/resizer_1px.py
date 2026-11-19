"""
縮小後の一辺が 1 画素でも動く Resizer (公式実装の resizer.py は変更せず、サブクラスで contributions だけを差し替える)。

元の Resizer.contributions は field_of_view と weights を np.squeeze するため、出力が 1 画素の次元
(32px で down_N=32) では出力側の次元まで潰れて IndexError になる。squeeze をやめて
[出力画素 x カーネル幅] の 2 次元を保つ。出力が 2 画素以上なら元の実装と重み・出力が完全に一致する
(patches/proposed-resizer-1px-output.diff と同じ修正。検証は diagnostics/verify_ilvr_eye.py の V5)。

contributions の中身は resizer.py (https://github.com/assafshocher/resizer by Assaf Shocher) からの写しで、
変更は squeeze をやめた 3 か所だけ。
"""

import numpy as np

from resizer import Resizer


class Resizer1px(Resizer):
    def contributions(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
        fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
        kernel_width *= 1.0 / scale if antialiasing else 1.0
        out_coordinates = np.arange(1, out_length + 1)
        shifted_out_coordinates = out_coordinates - (out_length - in_length * scale) / 2
        match_coordinates = shifted_out_coordinates / scale + 0.5 * (1 - 1 / scale)
        left_boundary = np.floor(match_coordinates - kernel_width / 2)
        expanded_kernel_width = np.ceil(kernel_width) + 2
        # 変更 1: squeeze しない
        field_of_view = np.int16(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1)
        weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)
        sum_weights = np.sum(weights, axis=1)
        sum_weights[sum_weights == 0] = 1.0
        weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)
        mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
        field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]
        non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
        # 変更 2, 3: squeeze せず、列の選択だけを行う
        weights = weights[:, non_zero_out_pixels[0]]
        field_of_view = field_of_view[:, non_zero_out_pixels[0]]
        return weights, field_of_view
