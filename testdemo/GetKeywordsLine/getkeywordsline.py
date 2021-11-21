# -*- coding: utf-8 -*-
# coding: utf-8

f = open('C:/Users/DUNKUN.XU/Desktop/sitalog_0011_20210916_1400.txt', encoding="utf-8")
w = open('C:/Users/DUNKUN.XU/Desktop/res.txt', 'w')
lines = f.readlines()
sum = 0
for line in lines:
    if "TEST" or "dunkun1" in line:
        w.write(line)
        sum += 1
        # if sum%20 == 0:
        #     w.write("-------------------\n")
        #     w.write("-------------------\n")
f.close()