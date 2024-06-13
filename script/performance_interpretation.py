#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:26:41 2023

@author: yan
"""

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import os
import time 
import seaborn as sns
import logging
import pandas
import sys
import sklearn.model_selection
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pickle
from scipy.stats import spearmanr,pearsonr
start_time=time.time()
import warnings
from sklearn.pipeline import Pipeline
import shap
warnings.filterwarnings('ignore')
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
                  python performance_interpretation.py TS2.tsv
                  """)
parser.add_argument("TS2_file", type=str, help="supplement table S2")
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-t","--target", type=str, default="AUC score", help="training target. (the metric scores). default: AUC score")
parser.add_argument("-T","--threshold", type=float, default=0, help="Threshold to filter based on training target. Keeping items with target > threshold. default: 0")
parser.add_argument("-s","--species", type=str, default=None, help="")
args = parser.parse_args()
TS2_file=args.TS2_file
output_file_name = args.output
species=args.species
target=args.target
threshold=args.threshold
try:
    os.mkdir(output_file_name)
except:
    overwrite=input("File exists, do you want to overwrite? (y/n)")
    if overwrite == "y":
        os.system("rm -r %s"%output_file_name)
        os.mkdir(output_file_name)
    elif overwrite =="n":
        output_file_name=input("Please give a new output file name:")
        os.mkdir(output_file_name)
    else:
        print("Please input valid choice..\nAbort.")
        sys.exit()
def Evaluation(output_file_name,y,predictions):
    #scores
    output=open(output_file_name+"/log.txt","a")
    output.write("*************\n")
    spearman_rho,spearman_p_value=spearmanr(y, predictions)
    pearson_rho,pearson_p_value=pearsonr(y, predictions)
    output.write("spearman correlation rho: "+str(spearman_rho)+"\n")
    output.write("spearman correlation p value: "+str(spearman_p_value)+"\n")
    output.write("pearson correlation rho: "+str(pearson_rho)+"\n")
    output.write("pearson correlation p value: "+str(pearson_p_value)+"\n")
    output.write("r2: "+str(sklearn.metrics.r2_score(y,predictions))+"\n")
    output.write("explained_variance_score score:"+str(sklearn.metrics.explained_variance_score(y, predictions))+"\n")
    output.write("Mean absolute error regression loss score:"+str(sklearn.metrics.mean_absolute_error(y, predictions))+"\n")
    y=np.array(y)
    
    # scatter plot
    plt.figure() 
    sns.set_palette("PuBu",2)
    gs = gridspec.GridSpec(3, 3)
    ax_main = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
    ax_main.scatter(y,predictions,edgecolors='white',alpha=0.8)
    ax_main.set(xlabel='True',ylabel='Predicted')
    ax_xDist.hist(y,bins=70,align='mid',alpha=0.7)
    ax_xDist.set(ylabel='count')
    ax_xDist.tick_params(labelsize=6,pad=2)
    ax_yDist.hist(predictions,bins=70,orientation='horizontal',align='mid',alpha=0.7)
    ax_yDist.set(xlabel='count')
    ax_yDist.tick_params(labelsize=6,pad=2)
    ax_main.text(0.55,0.03,"Spearman R: {0}".format(round(spearman_rho,2)),transform=ax_main.transAxes,fontsize=10)
    ax_main.text(0.55,0.10,"Pearson R: {0}".format(round(pearson_rho,2)),transform=ax_main.transAxes,fontsize=10)
    plt.savefig(output_file_name+"/evaluation.png",dpi=300)
    # plt.show()
    plt.close()
def self_encode(feature_value_list):#one-hot encoding for single nucleotide features
    classes=[str(i) for i in list(set(feature_value_list))]
    classes.sort()
    integer_encoded=np.zeros([len(feature_value_list),len(classes)],dtype=np.float64)
    for i in range(len(feature_value_list)):
        integer_encoded[i,classes.index(str(feature_value_list[i]))]=1
    return integer_encoded, classes

open(output_file_name + '/log.txt','w').write(time.asctime()+"\n")
open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)
performances=pandas.read_csv(TS2_file,sep='\t')
performances=performances[performances['split']!='random']
if species==None:
    performances=performances[performances[target]>=threshold]
elif "," in species:
    performances=performances[(performances[target]>=threshold)&(performances['species'].isin([s.replace("_"," ") if s!='extend_E_coli' else s for s in species.split(",") ]))]
elif species !='extend_E_coli':
    performances=performances[(performances[target]>=threshold)&(performances['species']==species.replace("_"," "))]
else:
    performances=performances[(performances[target]>=threshold)&(performances['species']==species)]
open(output_file_name + '/log.txt','a').write("Total number of samples in dataset: %s\n"% (performances.shape[0]))


performances['distance']=[10**(-1*i) for i in list(performances['distance'])]
case_label={0:'a',1:'b',2:'c',3:'d'}
performances['training_size']=np.array(performances['R1'])+np.array(performances['S1'])+np.array(performances['Rn'])+np.array(performances['Sn'])
drop_feature=['antibiotic','distance_rank','model','scheme','MOA','split']+['R1','S1','Rn','Sn']
if species!=None and ',' not in species:
    drop_feature.append('species')
for f in drop_feature:
    try:
        performances=performances.drop(f,axis=1)
    except:
        pass
performances=performances.dropna(subset=[target])
y=performances[target]
X=performances.drop(['AUC score', 'Precision', 'Recall','Balanced accuracy score'],axis=1)
catergorical_feat=['antibiotic','species','MOA','case','scheme','drug_class']
other_fea=[i for i in X.columns if i not in catergorical_feat]
X_num=X[other_fea]
open(output_file_name + '/log.txt','a').write("Features: %s\n"% (','.join(X.columns.values.tolist())))
for f in [i for i in catergorical_feat if i in X.columns]:
    integer_encoded, classes = self_encode(list(X[f]))
    X_num[classes]=integer_encoded
    open(output_file_name + '/log.txt','a').write("%s: %s\n"%(f,",".join(classes)))
X=X_num.copy()
X=X.astype(float)
headers=X.columns.values.tolist()
open(output_file_name + '/log.txt','a').write("Features: %s\n"% (','.join(X.columns.values.tolist())))
## random forest
from sklearn.ensemble import RandomForestRegressor
estimator=RandomForestRegressor(random_state=np.random.seed(111))
# taining
estimator = estimator.fit(np.array(X,dtype=float),np.array(y,dtype=float))
predictions=estimator.predict(X)
Evaluation(output_file_name,y,predictions)
print('finish training')
##SHAP
explainer=shap.TreeExplainer(estimator)
shap_values = explainer.shap_values(X,check_additivity=False)
values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':headers})
values.to_csv(output_file_name+"/shap_value_mean.csv",index=False,sep='\t')

plt.figure(figsize=(3,6))
shap.summary_plot(shap_values, X, plot_type="bar",show=False,color_bar=True,max_display=10)
plt.subplots_adjust(left=0.35, top=0.95)
plt.xticks(fontsize=12)
plt.savefig(output_file_name+"/shap_value_bar.svg")
plt.savefig(output_file_name+"/shap_value_bar.png",dpi=300)
plt.close()

plt.figure(figsize=(3,6))
shap.summary_plot(shap_values, X, plot_type="bar",show=False,color_bar=True,max_display=15)
plt.subplots_adjust(left=0.35, top=0.95)
plt.xticks(fontsize=12)
plt.savefig(output_file_name+"/shap_value_bar15.svg")
plt.savefig(output_file_name+"/shap_value_bar15.png",dpi=300)
plt.close()

for i in [10,15]:
    plt.figure(figsize=(3,6))
    shap.summary_plot(shap_values, X,show=False,max_display=i,alpha=0.1)
    plt.subplots_adjust(left=0.35, top=0.95,bottom=0.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize='small')
    plt.xticks(fontsize='small')
    plt.savefig(output_file_name+"/shap_value_%s.svg"%i)
    plt.savefig(output_file_name+"/shap_value_%s.png"%i,dpi=300)
    plt.close()     
    
    
