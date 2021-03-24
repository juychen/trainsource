import numpy as np
import csv
import os


def readCis(dirName, outRows):
    files = os.listdir(dirName)
    for file in files:  # 遍历文件夹
        fn = dirName + '/' + file

        with open(fn, "r") as f:
            # with open("testFile/reportCisplatin0100VAE2021-03-23-09-42-09.csv") as f:
            reader = csv.DictReader(f)  # DictReader为字典读取器

            index = 0

            for row in reader:
                index = index + 1
                # print(row)  # row为一个字典,可以使用key来进行访问某一行某一列中的数据
                if index == 1:
                    ds = row['batch_id']
                if index == 2:
                    w1 = row['cluster_res']
                if index == 3:
                    w2 = row['cluster_res']
                if index == 4:
                    w3 = row['cluster_res']
                if index == 5:
                    w4 = row['cluster_res']
                if index == 6:
                    w5 = row['cluster_res']
                if index == 7:
                    w6 = row['cluster_res']
                if index == 8:
                    w7 = row['cluster_res']

            writerRow = (file, ds, w1, w2, w3, w4, w5, w6, w7)
            outRows.append(writerRow)

    return outRows


def readBET(dirName, outRows):
    files = os.listdir(dirName)
    for file in files:  # 遍历文件夹
        fn = dirName + '/' + file

        with open(fn, "r") as f:
            reader = csv.DictReader(f)
            index = 0
            for row in reader:
                index = index + 1
                # print(row)
                if index == 3:
                    for c in row:
                        ds = row[c]
                if index == 42:
                    for c in row:
                        w1 = row[c]
                if index == 43:
                    for c in row:
                        w2 = row[c]
                if index == 44:
                    for c in row:
                        w3 = row[c]
                if index == 45:
                    for c in row:
                        w4 = row[c]
                if index == 46:
                    for c in row:
                        w5 = row[c]
                if index == 47:
                    for c in row:
                        w6 = row[c]
                if index == 48:
                    for c in row:
                        w7 = row[c]

            writerRow = (file, ds, w1, w2, w3, w4, w5, w6, w7)
            outRows.append(writerRow)

    return outRows


path_cis = '/users/PAS1475/qiren081/CCTS_DTL/trainsource-master0321/saved/upsampling/re_Cis'
path_bet = '/users/PAS1475/qiren081/CCTS_DTL/trainsource-master0321/saved/upsampling/re_BET'
outName = '/users/PAS1475/qiren081/CCTS_DTL/trainsource-master0321/saved/upsampling/up.csv'

# files  = os.listdir(path_heng)
# files2 = os.listdir(path_shu)

rows = []

rows = readCis(path_cis, rows)
rows = readBET(path_BET, rows)



headers = ['file_name', 'data_set', 'sens_pearson', 'resist_pearson', '1_pearson', '0_pearson', 'f1_score', 'auroc_score', 'ap_score']

with open(outName, 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)  # 单行写入，writerow和writerows中的参数只需要是可迭代的就行，并不一定是列表
    f_csv.writerows(rows)  # 多行写入
