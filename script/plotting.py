#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:12:58 2024

@author: yan
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import itertools
import seaborn as sns
import pandas
import sklearn.model_selection
import sklearn.metrics
from scipy.stats import spearmanr,pearsonr
from collections import defaultdict
import shap
import sys
import os
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib as mpl
import matplotlib.patches as mpatches
mpl.rcParams['figure.dpi'] = 400
import sklearn
import warnings
warnings.filterwarnings('ignore')
import scipy
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['font.sans-serif']='Arial'
mpl.rcParams['font.size']=14
mpl.rcParams['legend.title_fontsize']=10
mpl.rcParams['legend.fontsize']=10
mpl.rcParams['xtick.labelsize']=12
mpl.rcParams['ytick.labelsize']=12
#%%
'''
Figure 1
'''
import random
metrics_labels={'roc_auc_score':'AUC score','balanced_accuracy_score':'Balanced accuracy score',
                'precision_score':'Precision','recall_score':'Recall',
                'f1_score':'F1 score'
                }
# moa=pandas.read_csv("/home/yyu/Projects/ARP/seq/antibiotic_moa.csv",sep='\t')
moa=pandas.read_csv("/AMR_prediction/doc/antibiotic_moa.tsv",sep='\t')
moa_dict=dict()
for i in moa.index:
    moa_dict.update({moa['Antibiotic'][i]:moa['MOA'][i]})
metrics=list(metrics_labels.keys())[:4]
training_target='SIR'
folds=10
performances=defaultdict(list)

#### options
scheme='A'
random_option=False #random split
ratio_option=False # include ratio between R/S in sample split

if scheme=='A':
    result_folder='schemeA_result'
    anti_result_folder='schemeA_' # folder name is formated as, i.e. schemeA_amoxicillin
    table_name='schemeA_performance'
elif scheme=='B':
    result_folder='schemeB_result'
    anti_result_folder='schemeB_'
    table_name='schemeB_performance'
if random_option:
    table_name+="_random_split"
if ratio_option:
    result_folder='schemeA_ratio_result'
    anti_result_folder='schemeA_ratio_'
    table_name+="_ratio"

for species in ["extend_E_coli","Klebsiella pneumoniae","Salmonella enterica","Streptococcus pneumoniae","Staphylococcus aureus"]:
    corr=defaultdict(list)    
    # macos path
    output_file_name="AMR_prediction/data/ml/%s/%s"%(species.replace(" ","_"),result_folder)
    model_types = os.listdir(output_file_name)
    model_types=[i.strip() for i in model_types if "test_folder" not in i and anti_result_folder in i]
    model_types.sort()
    for model_type in model_types:
        split_file=model_type.split(anti_result_folder)[1]
        
        clade_distance_all=pandas.read_csv("AMR_prediction/clade_split/%s/clade_distance_%s.tsv"%(species.replace(" ","_"),split_file),sep='\t',index_col=0)
        
        scores=pandas.read_csv(output_file_name+"/"+model_type+"/lightgbm/iteration_scores.csv",sep='\t')
        antis=list(set(scores['antibiotic']))   
        antis.sort()
        split_file=pandas.read_csv("AMR_prediction/clade_split/%s/train_test_split_%s.tsv"%(species.replace(" ","_"),split_file),sep='\t',dtype={'genome_id':str})
        # 
        clades=list(set(split_file['split']))
        clades.sort()
        pal_clades=sns.color_palette("tab20",len(clades)).as_hex()
              
        for a in antis:
            clade_split_ratio_fold=pandas.read_csv(output_file_name+"/"+model_type+"/lightgbm/"+a+"/clade_split_ratio_fold.csv",sep='\t',dtype={'genome_id':str})
            training_clade=list(clade_distance_all.columns)[0]
            included_clade=list(set(clade_split_ratio_fold['clade']))
            included_clade.sort()
            pal=[pal_clades[i] for i in range(len(clades)) if clades[i] in included_clade]
            plot=defaultdict(list)
            test='lightgbm'
            for i in clade_split_ratio_fold.index:
                train_test_split_num=clade_split_ratio_fold['train_test_split'][i]
                if ratio_option:
                    ratio=clade_split_ratio_fold['ratio'][i] #v11
                
                ### recalculate the scores based on the balanced R/S
                try:
                    test_score=pandas.read_csv(output_file_name+"/"+model_type+"/"+test+"/"+a+"/test_predictions_%s.csv"%(train_test_split_num+1),sep='\t')
                    ## not random
                    if not random_option:
                        test_score=test_score[test_score['split']==clade_split_ratio_fold['clade'][i]] 
                    # random split: only include the sample types same as training
                    if random_option:
                        if clade_split_ratio_fold['R1'][i]==0:
                            test_score=test_score[~((test_score['split']!=clade_split_ratio_fold['clade'][i])&(test_score['y_test']==0))] 
                        elif clade_split_ratio_fold['S1'][i]==0:
                            test_score=test_score[~((test_score['split']!=clade_split_ratio_fold['clade'][i])&(test_score['y_test']==1))] 
                        elif clade_split_ratio_fold['Rn'][i]==0:
                            test_score=test_score[~((test_score['split']==clade_split_ratio_fold['clade'][i])&(test_score['y_test']==0))] 
                        elif clade_split_ratio_fold['Sn'][i]==0:
                            test_score=test_score[~((test_score['split']==clade_split_ratio_fold['clade'][i])&(test_score['y_test']==1))] 
                    test_R=test_score[test_score['y_test']==0].shape[0]
                    test_S=test_score[test_score['y_test']==1].shape[0]
                    if test_R> test_S:
                        for m in metrics:
                            m_scores=list()
                            for l in range(100):
                                R_samples=list(random.sample(list(test_score[test_score['y_test']==0].index),k=test_S))
                                inds=R_samples+list(test_score[test_score['y_test']==1].index)
                                y_test=np.array(test_score.loc[inds,'y_test'])
                                predictions_proba=np.array(test_score.loc[inds,'predictions_proba'])
                                predictions=[1 if i >= 0.5 else 0 for i in predictions_proba]
                                if m=='balanced_accuracy_score':
                                    m_scores.append(sklearn.metrics.balanced_accuracy_score(y_test, predictions))
                                elif m=='precision_score':
                                    m_scores.append(sklearn.metrics.precision_score(y_test, predictions))
                                elif m=='recall_score':
                                    m_scores.append(sklearn.metrics.recall_score(y_test, predictions))
                                elif m=='f1_score':
                                    m_scores.append(sklearn.metrics.f1_score(y_test, predictions))
                                elif m=='matthews_corrcoef':
                                    m_scores.append(sklearn.metrics.matthews_corrcoef(y_test, predictions))
                                elif m=='roc_auc_score':
                                    m_scores.append(sklearn.metrics.roc_auc_score(y_test, predictions_proba))
                            plot[metrics_labels[m]].append(np.mean(m_scores))
                            performances[metrics_labels[m]].append(np.mean(m_scores))
                    elif test_S > test_R:
                        for m in metrics:
                            m_scores=list()
                            for l in range(100):
                                S_samples=list(random.sample(list(test_score[test_score['y_test']==1].index),k=test_R))
                                inds=S_samples+list(test_score[test_score['y_test']==0].index)
                                y_test=np.array(test_score.loc[inds,'y_test'])
                                predictions_proba=np.array(test_score.loc[inds,'predictions_proba'])
                                predictions=[1 if i >= 0.5 else 0 for i in predictions_proba]
                                if m=='balanced_accuracy_score':
                                    m_scores.append(sklearn.metrics.balanced_accuracy_score(y_test, predictions))
                                elif m=='precision_score':
                                    m_scores.append(sklearn.metrics.precision_score(y_test, predictions))
                                elif m=='recall_score':
                                    m_scores.append(sklearn.metrics.recall_score(y_test, predictions))
                                elif m=='f1_score':
                                    m_scores.append(sklearn.metrics.f1_score(y_test, predictions))
                                elif m=='matthews_corrcoef':
                                    m_scores.append(sklearn.metrics.matthews_corrcoef(y_test, predictions))
                                elif m=='roc_auc_score':
                                    m_scores.append(sklearn.metrics.roc_auc_score(y_test, predictions_proba))
                            plot[metrics_labels[m]].append(np.mean(m_scores))
                            performances[metrics_labels[m]].append(np.mean(m_scores))
                            # print(np.mean(m_scores),score_a[m][train_test_split_num])
                    else:
                        scores=pandas.read_csv(output_file_name+"/"+model_type+"/"+test+"/iteration_scores.csv",sep='\t')
                        scores=scores.set_index('train_test_split')
                        for m in metrics:
                            plot[metrics_labels[m]].append(scores[m][train_test_split_num])
                            performances[metrics_labels[m]].append(scores[m][train_test_split_num])
                    
                except Exception as e:
                    print(e,test,a)
                    break
                if clade_split_ratio_fold['clade'][i]=='training':
                    clade='clade_0'
                else:
                    clade=clade_split_ratio_fold['clade'][i]
                performances['species'].append(species)
                performances['antibiotic'].append(a)
                performances['MOA'].append(moa_dict[a])
                performances['fold'].append(train_test_split_num)
                if ratio_option:
                    performances['ratio'].append(ratio)
                performances['case'].append(clade_split_ratio_fold['scenarios'][i])
                performances['R1'].append(clade_split_ratio_fold['R1'][train_test_split_num])
                performances['S1'].append(clade_split_ratio_fold['S1'][train_test_split_num])
                performances['Rn'].append(clade_split_ratio_fold['Rn'][train_test_split_num])
                performances['Sn'].append(clade_split_ratio_fold['Sn'][train_test_split_num])
                try:
                    if clade=='clade_0':
                        performances['distance'].append(0)
                    else:
                        performances['distance'].append(clade_distance_all[training_clade][clade])
                except Exception as e:
                    print(e, training_clade, clade,clade_distance_all)
                    sys.exit()
                performances['model'].append('LightGBM')
    performances_csv=pandas.DataFrame.from_dict(performances)
    performances_csv.to_csv("AMR_prediction/doc/%s.tsv"%table_name,sep='\t',index=False)
#%%
'''
# results for differences in Scenarios
'''
case_label={0:'a',1:'b',2:'c',3:'d'}
scheme_a=pandas.read_csv("AMR_prediction/doc/schemeA_performance.tsv",sep='\t')
scheme_b=pandas.read_csv("AMR_prediction/doc/schemeB_performance.tsv",sep='\t')
scheme_a['scheme']=['A']*scheme_a.shape[0]
scheme_b['scheme']=['B']*scheme_b.shape[0]
scheme_a['split']=['scheme']*scheme_a.shape[0]
scheme_b['split']=['scheme']*scheme_b.shape[0]
for i in scheme_a.index:
    scheme_a.at[i,'case']='A-'+case_label[scheme_a['case'][i]]
    if scheme_a['species'][i]=='extend_E_coli':
        scheme_a.at[i,'species']='Escherichia coli'
for i in scheme_b.index:
    scheme_b.at[i,'case']='B-'+case_label[scheme_b['case'][i]]
    if scheme_b['species'][i]=='extend_E_coli':
        scheme_b.at[i,'species']='Escherichia coli'
scheme_a_random=pandas.read_csv("AMR_prediction/doc/schemeA_performance_random_split.tsv",sep='\t')
scheme_b_random=pandas.read_csv("AMR_prediction/doc/schemeB_performance_random_split.tsv",sep='\t')
scheme_a_random['scheme']=['A']*scheme_a_random.shape[0]
scheme_b_random['scheme']=['B']*scheme_b_random.shape[0]
scheme_a_random['split']=['random']*scheme_a_random.shape[0]
scheme_b_random['split']=['random']*scheme_b_random.shape[0]

df=pandas.concat([scheme_a,scheme_b,scheme_a_random,scheme_b_random]).reset_index(drop=True)

count=pandas.read_csv("AMR_prediction/doc/antibiotic_mechanism_count.tsv",sep='\t')
count=count.set_index('antibiotic')
for i in df.index:
    anti=df['antibiotic'][i]
    for j in ['drug_class']:
        df.at[i,j]=count[j][anti]
df.to_csv("AMR_prediction/doc/TS2.tsv",sep='\t',index=False)
#%%
'''
Figure 1
'''
df=pandas.read_csv("AMR_prediction/doc/TS2.tsv",sep='\t')
sns.set_palette('Pastel2')
sns.set_style('whitegrid')
plt.figure(figsize=(4,4))
sns.boxplot(data=df[df['scheme']=='A'],x='case',y='AUC score',order=list(range(4)),palette=['lightgrey','darkgrey'],hue='split')
plt.xlabel("Scenarios")
plt.xticks(list(range(4)),['a','b','c','d'],fontsize=12,weight='bold')
plt.title('Scheme A')
plt.legend(title="")
plt.savefig("AMR_prediction/doc/figures/1D.svg")
plt.show()
plt.close()

plt.figure(figsize=(4,4))
sns.boxplot(data=df[df['scheme']=='B'],x='case',y='AUC score',order=list(range(2)),palette=['lightgrey','darkgrey'],hue='split')
plt.xticks(list(range(2)),['a','b'],fontsize=12,weight='bold')
plt.xlabel("Scenarios")
plt.title('Scheme B')
plt.legend(title="")
plt.savefig("AMR_prediction/doc/figures/1F.svg")
plt.show()
plt.close()

df.loc[df[(df['scheme']=='B')&(df['split']=='random')&(df['case']==0)].index,'case']=10
df.loc[df[(df['scheme']=='B')&(df['split']=='random')&(df['case']==1)].index,'case']=11

plt.figure(figsize=(5,4))
ax=sns.boxplot(data=pandas.melt(df[df['scheme']=='B'], id_vars=['case'], value_vars=['Precision','Recall']),hue='case',x='variable',y='value',hue_order=[0,1,10,11],palette='Pastel2')
plt.xlabel("")
plt.ylabel('Score')
group_handles, labels = ax.get_legend_handles_labels()
plt.legend(group_handles,['B-a','B-b','B-a-random','B-b-random'],title='',ncol=1,bbox_to_anchor=(1,1))
plt.savefig("AMR_prediction/doc/figures/1G.svg")
plt.show()
plt.close()
#%%
'''
Figure 2
'''
#2B
plt.rcParams['text.usetex'] = False
case_label={0:'a',1:'b',2:'c',3:'d'}
scheme_a=pandas.read_csv("AMR_prediction/doc/schemeA_performance.tsv",sep='\t')
scheme_a=scheme_a[scheme_a['model']=='LightGBM']
scheme_b=pandas.read_csv("AMR_prediction/doc/schemeB_performance.tsv",sep='\t')
scheme_b=scheme_b[scheme_b['model']=='LightGBM']
for i in scheme_a.index:
    scheme_a.at[i,'case']='A-'+case_label[scheme_a['case'][i]]
    if scheme_a['species'][i]=='extend_E_coli':
        scheme_a.at[i,'species']='Escherichia coli'
for i in scheme_b.index:
    scheme_b.at[i,'case']='B-'+case_label[scheme_b['case'][i]]
    if scheme_b['species'][i]=='extend_E_coli':
        scheme_b.at[i,'species']='Escherichia coli'
scheme_a['scheme']=['A']*scheme_a.shape[0]
scheme_b['scheme']=['B']*scheme_b.shape[0]
df=pandas.concat([scheme_a,scheme_b])
species=["Escherichia coli","Klebsiella pneumoniae","Salmonella enterica","Streptococcus pneumoniae","Staphylococcus aureus"]
fig, axes = plt.subplots(len(species),sharex=True,figsize=(6,6))
for i in range(len(species)):
    sns.distplot(df[df['species']==species[i]]['AUC score'],ax=axes[i],kde=False,bins=70,color='darkgrey')
    axes[i].set_title('$\it{0}$ $\it{1}$'.format(species[i].split(" ")[0],species[i].split(" ")[1]),pad=2,fontsize=12)
    axes[i].axvline(np.median(df[df['species']==species[i]]['AUC score']),color='red',linestyle='--')
for i in range(len(species)):
    axes[i].set_xlabel('')
    axes[i].tick_params(labelsize=10,pad=0.5)
plt.subplots_adjust(hspace=0.4)
plt.xlim(0,1)
fig.text(0.01, 0.5, 'Count', va='center',rotation='vertical',fontsize=12)
plt.xlabel("AUC socre",fontsize=12)
plt.show()
plt.close()

#2C
df=pandas.read_csv("AMR_prediction/doc/TS2.tsv",sep='\t')
df=df[df['split']!='random']
species=["Escherichia coli","Klebsiella pneumoniae","Salmonella enterica","Streptococcus pneumoniae","Staphylococcus aureus"]
MOA=['penam','carbapenem','cephalosporin','fluoroquinolone antibiotic','others']
fig, axes = plt.subplots(len(MOA),sharex=True,figsize=(6,6))
for i in range(len(MOA)):
    if MOA[i] !='others':
        sns.distplot(df[df['drug_class']==MOA[i]]['AUC score'],ax=axes[i],kde=False,bins=70,color='darkgrey')
        axes[i].set_title(MOA[i],pad=2,fontsize=12)
        axes[i].axvline(np.median(df[df['drug_class']==MOA[i]]['AUC score']),color='red',linestyle='--')
    else:
        sns.distplot(df[~df['drug_class'].isin(MOA[:i])]['AUC score'],ax=axes[i],kde=False,bins=70,color='darkgrey')
        axes[i].set_title(MOA[i],pad=2,fontsize=12)
        axes[i].axvline(np.median(df[~df['drug_class'].isin(MOA[:i])]['AUC score']),color='red',linestyle='--')
for i in range(len(MOA)):
    axes[i].set_xlabel('')
    axes[i].tick_params(labelsize=10,pad=0.5)
plt.subplots_adjust(hspace=0.4)
plt.xlim(0,1)
plt.xlabel("AUC socre",fontsize=12)
fig.text(0.01, 0.5, 'Count', va='center',rotation='vertical',fontsize=12)
plt.show()
plt.close()
#%%
'''
#Figure S1
'''

'''
#S1A plot the species and no. genomes range of antibiotics
'''
species=["Escherichia coli","Klebsiella pneumoniae","Salmonella enterica","Streptococcus pneumoniae","Staphylococcus aureus"]
sample_size=pandas.read_csv("AMR_prediction/doc/clade_wise_sample_size.tsv",sep='\t')
df=sample_size.groupby(['species','antibiotic']).sum()
df=df.reset_index(level=['species','antibiotic'])
print(df)
plt.figure(figsize=(6,3))
for i in df.index:
    if df['species'][i]=='extend_E_coli':
        df.at[i,'species']='Escherichia coli'
    df.at[i,'species']='$\it{0}$\n$\it{1}$'.format(df['species'][i].split(" ")[0],df['species'][i].split(" ")[1])
sns.boxplot(data=df,x='species',y='size',color='lightgrey',order=['$\it{0}$\n$\it{1}$'.format(species[i].split(" ")[0],species[i].split(" ")[1]) for i in range(len(species))])
plt.xlabel("")
plt.ylabel("No. genomes")
plt.xticks(rotation=30)
# plt.savefig("AMR_prediction/figures/S1A.svg")
plt.show()
plt.close()


'''
#S1B plot the species and no. antibiotic in each MOA
'''
species=["Escherichia coli","Klebsiella pneumoniae","Salmonella enterica","Streptococcus pneumoniae","Staphylococcus aureus"]
moa=pandas.read_csv("AMR_prediction/doc/antibiotic_moa.tsv",sep='\t')
moa_dict=dict()
for i in moa.index:
    moa_dict.update({moa['Antibiotic'][i]:moa['MOA'][i]})
sample_size=pandas.read_csv("/Users/yan/Projects/AMR_prediction/data/dataset_exploration/clade_wise_sample_size.tsv",sep='\t')
df=sample_size.groupby(['species','antibiotic']).sum()
df=df.reset_index(level=['species','antibiotic'])
for i in df.index:
    if df['species'][i]=='extend_E_coli':
        df.at[i,'species']='Escherichia coli'
for i in df.index:
    df.at[i,'MOA']=moa_dict[df['antibiotic'][i]]
df.to_csv("AMR_prediction/doc/TS1.tsv",sep='\t',index=False)
df=df.groupby(['species','MOA']).count()
df=df.reset_index(level=['species','MOA'])


MOA=list(set(df['MOA']))
MOA.sort()
moa_dict=dict()
for m in MOA:
    df_m=df[df['MOA']==m]
    df_m=df_m.set_index('species')
    for s in species:
        if s not in df_m.index:
            df_m.at[s,'antibiotic']=0
    df_m=df_m.loc[species]
    moa_dict[m]=np.array(list(df_m['antibiotic']))
    
sns.set_palette("Pastel2")
fig, ax = plt.subplots(figsize=(6,3))
bottom = np.zeros(len(set(df['species'])))
for boolean, weight_count in moa_dict.items():
    p = ax.bar(species, weight_count, label=boolean, bottom=bottom)
    bottom += weight_count
plt.xticks(list(range(len(species))),['$\it{0}$\n$\it{1}$'.format(species[i].split(" ")[0],species[i].split(" ")[1]) for i in range(len(species))],rotation=30)
plt.ylabel("No. antibiotics")
plt.legend(fontsize=12)
plt.ylim(0,12)
plt.show()
plt.close()

#%%
'''
Figure S2-S6
S2-6A please see itol_visualization folder
'''
'''
S2-6B
check phylogenetic distances between clades
'''
pal=sns.color_palette('Pastel2').as_hex()
for species in ["extend_E_coli","Klebsiella pneumoniae","Salmonella enterica","Streptococcus pneumoniae","Staphylococcus aureus"]: #
    clades=open("clade_split/%s/clades.txt"%(species.replace(" ","_")),'r')
    clades_file=clades.readlines()
    clade_dict=defaultdict(list)
    for c in clades_file:
        clade_dict[c.split("\t")[1].strip()].append(str(c.split("\t")[0].replace("#","_")))
    clades=list(set(clade_dict.keys()))
    clades.sort()
    clades=[i for i in clades if i != 'val']
    print(clades)
    combs=list(itertools.combinations(clades, 2)) ##pair-wise combination of included STs
        
    distance_file=pandas.read_csv("%s/phylogeny_distance.tsv"%(species.replace(" ","_")),sep='\t', dtype=str,low_memory = True)
    distance=distance_file.rename(columns={distance_file.columns.values.tolist()[0]:'genome_id'})
    
    distance=distance.set_index('genome_id')
    distance=distance.astype(float)
    plt.figure()
    for c in combs:
        if c[0]=='training' or c[1]=='training':
            inds_1=clade_dict[c[0]]
            inds_2=clade_dict[c[1]]
            
            distance_c=np.array(distance.loc[inds_1,inds_2],dtype=float).flatten()
            
            #     ##dist plot
            if c[0]=='training':
                label='clade '+str(int(c[1].split("_")[1])+1)
                color=pal[clades.index(c[1])]
            else:
                label='clade '+str(int(c[0].split("_")[1])+1)
                color=pal[clades.index(c[0])]
            sns.distplot(distance_c,label=label,color=color)
        
    plt.xlabel("Distances between genomes of clade 1 and other clades")
    plt.legend(title="")
    if species=='extend_E_coli':
        plt.title('$\itEscherichia$ $\itcoli$',fontsize=12)
    else:
        plt.title('$\it{0}$ $\it{1}$'.format(species.split(" ")[0],species.split(" ")[1]),fontsize=12)
    plt.show()
    plt.close()
#%%
'''
Figure S7
'''

'''
S7A
'''
df=pandas.read_csv("AMR_prediction/doc/TS2.tsv",sep='\t')
plt.figure(figsize=(5,4))
ax=sns.boxplot(data=pandas.melt(df[(df['scheme']=='A')&(df['split']!='random')], id_vars=['case'], value_vars=['Precision','Recall']),hue='case',x='variable',y='value',hue_order=[0,1,2,3],palette='Pastel2')
plt.xlabel("")
plt.ylabel('Score')
group_handles, labels = ax.get_legend_handles_labels()
plt.legend(group_handles,['a','b','c','d'],title='',ncol=1,bbox_to_anchor=(1,1))
plt.show()
plt.close()

'''
S7BC
'''
ratio=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
performances=pandas.read_csv("AMR_prediction/doc/schemeA_performance_ratio.tsv",sep='\t')
df=pandas.read_csv("AMR_prediction/doc/schemeA_performance.tsv",sep='\t')
df=df[df['model']=='LightGBM']

for f in ['Precision','Recall']:
    for c in [0]:
        scheme=performances[performances['case']==c]
        if c==0:
            A=df[df['case'].isin([0,1])]
            A.loc[A[A['case']==0].index,'ratio']=1
            A.loc[A[A['case']==1].index,'ratio']=0
        if c==1:
            A=df[df['case'].isin([2,3])]
            A.loc[A[A['case']==2].index,'ratio']=1
            A.loc[A[A['case']==3].index,'ratio']=0
            
        sns.boxplot(data=pandas.concat([scheme,A],axis=0).reset_index(drop=True),x='ratio',y=f,color='lightgrey',order=ratio)
        plt.xticks(range(len(ratio)),[i*100 for i in ratio])
        if c==0:
            plt.xlabel("% R in scenario a")
            plt.title('Scheme A (a to b)')
        else:
            plt.xlabel("% R in scenario c")
            plt.title('Scheme A (c to d)')
        plt.tight_layout()
        plt.show()
        plt.close()

#%%
'''
Figure S8
'''
plot=defaultdict(list)
for species in ["extend_E_coli","Klebsiella pneumoniae","Salmonella enterica","Streptococcus pneumoniae","Staphylococcus aureus"]:
    corr=defaultdict(list)    
    output_file_name="AMR_prediction/data/ml/%s/individual_clade_result"%(species.replace(" ","_"))

    model_types=os.listdir(output_file_name)
    model_types=[i.strip() for i in model_types if "test_folder" not in i and 'individual_clade_' in i]
    print(species,model_types)
    for model_type in model_types:
        anti=model_type.split("individual_clade_")[1]
        split_file=pandas.read_csv("AMR_prediction/data/ml/%s/train_test_split_tables/train_test_split_%s.tsv"%(species.replace(" ","_"),anti),sep='\t',dtype={'genome_id':str})

        clades=list(set(split_file['split']))
        clades.sort()
        test='lightgbm'
        ### features in training clade
        for c in ['training']+[i for i in clades if i!='training']:
            try:
                shap_df=pandas.read_csv(output_file_name+"/"+model_type+"/"+test+"/"+anti+"/shap_value_mean_class0_%s.csv"%c,sep='\t',dtype={"shap_values":float})
                shap_df=shap_df.sort_values(by='shap_values',ascending=False)
                value_sum=sum(shap_df['shap_values'])
                value=0
                top_N=0
                while value < value_sum*0.5:
                    top_N+=1
                    value=sum(list(shap_df['shap_values'])[:top_N])
                
                fea_training=list(shap_df['features'])[:top_N]
                fea_training_top10=list(shap_df['features'])[:10]
                com_clade=c
                
                break
            except Exception as e:
                continue
        if len(fea_training)==0:
            continue
        for c in clades:
            if c==com_clade:
                continue
            
            try:
                shap_df=pandas.read_csv(output_file_name+"/"+model_type+"/"+test+"/"+anti+"/shap_value_mean_class0_%s.csv"%c,sep='\t',dtype={"shap_values":float})
                shap_df=shap_df.sort_values(by='shap_values',ascending=False)
                value_sum=sum(shap_df['shap_values'])
                value=0
                top_N=0
                while value < value_sum*0.5:
                    top_N+=1
                    value=sum(list(shap_df['shap_values'])[:top_N])
                overlap_count=len([i for i in fea_training if i in list(shap_df['features'])[:top_N]])/len(fea_training)*100
                overlap_count_top10=len([i for i in fea_training_top10 if i in list(shap_df['features'])[:10]])/len(fea_training_top10)*100
            except FileNotFoundError:
                overlap_count=np.nan
                continue
            if species=='extend_E_coli':
                plot['species'].append('Escherichia\ncoli')
            else:
                plot['species'].append('{0}\n{1}'.format(species.split(" ")[0],species.split(" ")[1]))
            plot['count'].append(overlap_count)
            plot['top_N'].append(top_N)
            plot['count_top10'].append(overlap_count_top10)

plot=pandas.DataFrame.from_dict(plot)
sns.set_style('white')
## S8A
species=["Escherichia coli","Klebsiella pneumoniae","Salmonella enterica","Streptococcus pneumoniae","Staphylococcus aureus"]
ax=sns.boxplot(data=plot,x='species',y='count_top10',order=['{0}\n{1}'.format(i.split(" ")[0],i.split(" ")[1]) for i in species],palette='Pastel2')
labels = ax.get_yticklabels()
for lbl in labels:
    lbl.set_style('italic')
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Overlapping features\nin top 10 SHAP features (%)")
plt.show()
plt.close()

#S8B
sns.boxplot(data=plot,x='species',y='count',order=['{0}\n{1}'.format(i.split(" ")[0],i.split(" ")[1]) for i in species],palette='Pastel2')
plt.xticks(rotation=30)
plt.xlabel("")
plt.ylabel("Overlapping features\nin 50% summed SHAP features (%)")
plt.show()
plt.close()
#S8C
plt.figure(figsize=(3,6))
sns.swarmplot(data=plot,y='top_N',color='grey')
plt.ylabel("The number of 50% summed SHAP features")
plt.xlabel("")
plt.show()
plt.close()
#%%
'''
Figure S9
'''

plot=defaultdict(list)
for species in ["extend_E_coli","Klebsiella pneumoniae","Salmonella enterica","Streptococcus pneumoniae","Staphylococcus aureus"]:
    corr=defaultdict(list)    
    output_file_name="/AMR_prediction/data/ml/%s/individual_clade_result"%(species.replace(" ","_"))
    snp_genename=open("AMR_prediction/data/ml/%s/core_alignment_header.embl"%(species.replace(" ","_")),"r")
    snp_genename_file=snp_genename.readlines()
    snp_genename=dict()
    for l in range(len(snp_genename_file)):
        line=snp_genename_file[l]
        if 'feature' in line:
            genome_range=line.split("feature")[1].replace(" ","")
            left=int(genome_range.split(".")[0])
            right=int(genome_range.split(".")[-1])
            label=snp_genename_file[l+1].split("label=")[1].strip()
            locus_tag=snp_genename_file[l+2].split("locus_tag=")[1].strip()
            snp_genename.update({tuple([left,right]):[label,locus_tag]})
    # print(snp_genename)
    genename_anno_file=pandas.read_csv("/AMR_prediction/data/ml/%s/gene_presence_absence_name_to_anno.tsv"%(species.replace(" ","_")),sep=',')
    genename_anno=dict()
    # print(genename_anno_file.columns)
    for i in genename_anno_file.index:
        genename_anno.update({genename_anno_file['Gene'][i]:[genename_anno_file['Non-unique Gene name'][i],genename_anno_file['Annotation'][i]]})


    model_types=os.listdir(output_file_name)
    model_types=[i.strip() for i in model_types if "test_folder" not in i and 'individual_clade_' in i]
    print(species,model_types)
    # model_types=['v10_ciprofloxacin']
    for model_type in model_types:
        anti=model_type.split("individual_clade_")[1]
        if anti !='ciprofloxacin':
            continue
        split_file=pandas.read_csv("AMR_prediction/data/ml/%s/train_test_split_tables/train_test_split_%s.tsv"%(species.replace(" ","_"),anti),sep='\t',dtype={'genome_id':str})

        # 
        clades=list(set(split_file['split']))
        clades.sort()
        test='lightgbm'
        ### features in training clade
        for c in ['training']+[i for i in clades if i!='training']:
            try:
                print(c)
                shap_df=pandas.read_csv(output_file_name+"/"+model_type+"/"+test+"/"+anti+"/shap_value_mean_class0_%s.csv"%c,sep='\t',dtype={"shap_values":float})
                shap_df=shap_df.sort_values(by='shap_values',ascending=False)
                value_sum=sum(shap_df['shap_values'])
                value=0
                top_N=0
                while value < value_sum*0.5:
                    top_N+=1
                    value=sum(list(shap_df['shap_values'])[:top_N])
                
                fea_training=list(shap_df['features'])[:top_N]
                fea_training_top10=list(shap_df['features'])[:10]
                com_clade=c
                shap_df=shap_df.iloc[:10,:]
                for i in shap_df.index:
                    if shap_df['features'][i] in genename_anno.keys():
                        shap_df.at[i,'features']=shap_df['features'][i]+"\n("+genename_anno[shap_df['features'][i]][1]+")"
                    else:
                        loc=int(shap_df['features'][i].split("_")[0])
                        ori=shap_df['features'][i].split("_")[1]
                        mut=shap_df['features'][i].split("_")[2]
                        mut=mut.replace("x","*")
                        for key in snp_genename.keys():
                            if loc>=key[0] and loc <= key[1]:
                                gene=snp_genename[key][1]
                                shap_df.at[i,'features']=gene+" ("+str(loc-key[0]+1)+","+ori+">"+mut+")"+"\n("+genename_anno[gene][1]+")"
                                break
                # print(shap_df)
                plt.figure(figsize=(3,5))
                sns.barplot(data=shap_df,x='shap_values',y='features',color='dodgerblue')
                plt.xlabel("mean(|SHAP value|)")
                plt.ylabel("")
                if species=='extend_E_coli':
                    title='$\it{\ Escherichia}$ $\it{\ coli}$'
                else:
                    title='$\it{\ %s}$ $\it{\ %s}$'%(species.split(" ")[0],species.split(" ")[1])
                if c=='training':
                    plt.title(title+"\n(%s %s)"%(anti,'clade 1'))
                else:
                    plt.title(title+"\n(%s %s)"%(anti,'clade '+str(int(c.split("_")[1])+1)))
                plt.show()
                plt.close()
                
            except Exception as e:
                continue