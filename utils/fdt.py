#!/usr/bin/env python

import numpy as np
import math


class FDT:
    def __init__(self, src):
        self.src_row = src.shape[0]
        self.src_col = src.shape[1]
        self.volume_size = self.src_col * self.src_row
        self.src = src

    def get_link_dist(self, row, col, row1, col1, data):
        link_dist = data[row*self.src_col+col]
        link_dist = link_dist+data[row1*self.src_col+col1]
        link_dist = link_dist/2
        return link_dist

    def cal_fdt(self, fdt=False, uint8=True):
        size_row = 1
        size_col = 1
        # Euclid distance
        size_row_col = math.sqrt(size_row * size_row + size_col * size_col)
        dist_val_pos = [size_row, size_row_col, size_col, size_row_col, size_row, size_row_col, size_col,
                        size_row_col, size_col, size_row_col]
        dist_val_neg = [size_row, size_row_col, size_col, size_row_col, size_row, size_row_col, size_col,
                        size_row_col, size_col, size_row_col]

        start_element_1 = 0
        end_element_1 = 4
        start_element_2 = 4
        end_element_2 = 8

        nbor_pos = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]]
        nbor_neg = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]]

        data_dt = np.array([0] * self.volume_size)
        data_mark = np.array([0] * self.volume_size)
        # 设立data_mark数组，反射边缘为0，反射区域为1
        for i in range(0, self.src_row, 1):
            for j in range(0, self.src_col, 1):
                if i == 0:
                    data_mark[i * self.src_col + j] = 0
                elif j == 0:
                    data_mark[i * self.src_col + j] = 0
                elif i == self.src_row - 1:
                    data_mark[i * self.src_col + j] = 0
                elif j == self.src_col - 1:
                    data_mark[i * self.src_col + j] = 0
                else:
                    data_mark[i * self.src_col + j] = 1

        # 把图像灰度信息录入data数组
        data = self.src.copy().flatten()

        # 划定反射区域,0为背景，极大为反射域
        data_dt[data != 0] = 65535

        # 记录反射点对应坐标，-1为反射域
#         ref_idx = np.array([-1] * self.volume_size)
        
        
        iteration = 0
        flag_change = 1

        while iteration < 5000 and flag_change == 1:
            flag_change = 0
            iteration = iteration + 1
            for row in range(self.src_row):
                for col in range(self.src_col):
                    if data_mark[row * self.src_col + col] == 1:
                        cur_dist = data_dt[row * self.src_col + col]
                        for ei in range(start_element_1, end_element_1, 1):
                            x = col + nbor_pos[ei][0]
                            y = row + nbor_pos[ei][1]
                            if (x >= 0) and (x < self.src_col) and (y >= 0) and (y < self.src_row):
                                link_dist = dist_val_pos[ei]
                                if fdt:
                                    link_dist = link_dist * self.get_link_dist(row, col, y, x, data)
                                next_dist = data_dt[y * self.src_col + x] + link_dist
                                if next_dist < cur_dist:
                                    flag_change = 1
                                    data_dt[row * self.src_col + col] = next_dist
                                    # 记录对应反射点
#                                     ref_idx[row * self.src_col + col] = y * self.src_col + x
                                    cur_dist = next_dist

            # backward scan
            for row in range(self.src_row - 1, -1, -1):
                for col in range(self.src_col - 1, -1, -1):
                    if data_mark[row * self.src_col + col] == 1:
                        cur_dist = data_dt[row * self.src_col + col]
                        for ei in range(start_element_2, end_element_2, 1):
                            x = col + nbor_neg[ei][0]
                            y = row + nbor_neg[ei][1]
                            if (x >= 0) and (x < self.src_col) and (y >= 0) and (y < self.src_row):
                                link_dist = dist_val_neg[ei]
                                if fdt:
                                    link_dist = link_dist * self.get_link_dist(row, col, y, x)
                                next_dist = data_dt[y * self.src_col + x] + link_dist
                                if next_dist < cur_dist:
                                    flag_change = 1
                                    data_dt[row * self.src_col + col] = next_dist
                                    # 记录对应反射点
#                                     ref_idx[row * self.src_col + col] = y * self.src_col + x
                                    cur_dist = next_dist

        # 寻找data_dt最大值
        maxint = 0
        for i in range(self.volume_size):
            if data_dt[i] == 65535:
                data_dt[i] = 1

        for i in range(self.volume_size):
            if data_dt[i] > maxint:
                maxint = data_dt[i]

        for i in range(self.volume_size):
            if maxint <= 65535:
                data[i] = data_dt[i] * 255 / maxint
                # data[i]=data_dt[i]
            else:
                data[i] = ((data_dt[i] * 65000) / maxint)

        data = np.array(data)
        np.set_printoptions(threshold=np.inf)
        data = np.reshape(data, (self.src_row, self.src_col))
        data = data.astype(np.uint8)

        if uint8:
            return data.reshape((self.src_row, self.src_col))
#             return data.reshape((self.src_row, self.src_col)), ref_idx.reshape((self.src_row, self.src_col))
        else:
            return data_dt.reshape((self.src_row, self.src_col))
#             return data_dt.reshape((self.src_row, self.src_col)), ref_idx.reshape((self.src_row, self.src_col))
