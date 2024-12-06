#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:42:14 2021

@author: aouyed
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def rmse_plot(df0,  var):
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


    df = df0.loc[(df0.model=='rf') & (df0.tele==True)]
    ax.plot(df['lead'], df[var], '-o', label='rf, tele')
    
    df = df0.loc[(df0.model=='rf') & (df0.tele==False)]
    ax.plot(df['lead'], df[var], '-o', label='rf, no tele')


    df = df0.loc[(df0.model=='linear') & (df0.tele==True)]
    ax.plot(df['lead'], df[var], '-o', label='linear, tele')
    
    df = df0.loc[(df0.model=='linear') & (df0.tele==False)]
    ax.plot(df['lead'], df[var], '-o', label='linear, no tele')

    ax.legend(frameon=None)
    ax.set_xlabel("lead week")
    ax.set_ylabel("rmse")
    ax.set_title(var)
    directory = '../data/processed/'+var
    plt.savefig(directory+'.png', bbox_inches='tight', dpi=300)
    plt.close()
def main():
    df=pd.read_csv('../data/processed/df_results.csv')
    print(df)
    rmse_plot(df, 'P')
    rmse_plot(df, 'T')
    
    
    
if __name__ == '__main__':
    main()