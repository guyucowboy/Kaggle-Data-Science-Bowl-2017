#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 17:24:45 2017

@author: anindya
"""
import csv
import shutil

def main():
    
    nonCancerRemovedCount = 0
    with open('./Data/stage1_labels.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['id'], row['cancer'])
            if row['cancer'] == '0':
                if nonCancerRemovedCount < 300:
                    nonCancerRemovedCount = nonCancerRemovedCount + 1
                    shutil.rmtree('/Data/sample_images_copy/'+row['id'])
                    print(row['id'])

if __name__ == '__main__':
    main()
