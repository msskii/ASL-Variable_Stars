#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:11:46 2023

@author: Gian
"""
import numpy as np

def time_extract(date_list):
    ''' from a list of dates of format yr_month_day_h_min_s we want 
    only extract h_min_s and convert to secs'''
    h_m_s_list = date_list.copy()
    h = date_list.copy()
    m = date_list.copy()
    s = date_list.copy()
    hstr = date_list.copy()
    mstr = date_list.copy()
    sstr = date_list.copy()
    for i in np.arange(h_m_s_list.size):
        h_m_s_list[i] = date_list[i].lstrip("2023-01-18_")
        h_m_s_list[i] = date_list[i][11:]
        hstr[i],mstr[i],sstr[i] = h_m_s_list[i].split('_')
        h[i],m[i],s[i] = int(hstr[i]),int(mstr[i]),int(sstr[i])
    return 60*60*(h-2) + 60*(m-56) + (s-56)