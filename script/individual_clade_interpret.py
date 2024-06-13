#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:34:42 2019

@author: yanying
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
import os
import time 
import pandas
import sklearn.model_selection
import sklearn.metrics
from collections import defaultdict
import shap
import sys
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chisquare
from sklearn.feature_selection import VarianceThreshold
import warnings
import lightgbm as lgb
from hyperopt import hp, tpe, Trials
import hyperopt
from hyperopt.fmin import fmin 
from hyperopt.early_stop import no_progress_loss
import pickle
import random
warnings.filterwarnings('ignore')
start_time=time.time()
nts=['A','T','C','G']
items=list(itertools.product(nts,repeat=2))
dinucleotides=list(map(lambda x: x[0]+x[1],items))
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to train lightGBM models to predict antibiotic resistance with k-mer, snp sites, or gene presence/absence.
Train and test were split based on resistance or susceptible in each ST.  

Example: python individual_clade_interpret.py gene_presence_absence.Rtab,snp_sites.vcf metadata_checkm.csv -t pres+vcf -s train_test_split_amoxicillin.tsv -A amoxicillin -o lightgbm -P params_saved 
                  """)
parser.add_argument("data_file", help="precomputed CSV files for kmer or vcf or pres or pres,vcf")
parser.add_argument("metadata", help="metadata CSV file")
parser.add_argument("-t","--input_type",type=str,default='pres+vcf', help="""
                    input CSV file type:
                        vcf: SNP calling file from snp-sites
                        kmer: N genomes * K kmers table
                        pres: K genes * N genomes table
                        pres+vcf: pres_csv,vcf_csv
                    """)
parser.add_argument("-A","--antibiotics", type=str, default="amoxicillin", help="The antibiotics for testing, should be the same as the training test split file. default: amoxicillin")
parser.add_argument("-s","--split_file", type=str, default='train_test_split.tsv', help="file to specify samples for train-test split (which clade the genomes are in). Two columns are requirred: genome_id and split, default: train_test_split.tsv")

parser.add_argument("-f","--folds", type=int, default=10, help="Repeat N times for each ratio of R:S samples. default: 10")
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-P", "--params_output", type=str,default=None, help="output folder name to save optimized params. default: None")
parser.add_argument("-r", "--random_seed",type=int, default=111, help="random seed. default: 111")

args = parser.parse_args()
data_file=args.data_file
metadata=args.metadata
input_type=args.input_type
output_file_name = args.output

antis=args.antibiotics
antis=antis.split(",")
split_file=args.split_file
split_file=pandas.read_csv(split_file,sep='\t',dtype={'genome_id':str})

params_output=args.params_output
random_seed=args.random_seed
folds=args.folds
test_size=0.2
estimator_type='classifier'

training_target='SIR'
objective='binary'
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
if params_output!=None:
    try:
        os.mkdir(params_output)
    except FileExistsError:
        pass


def is_number(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

def convert_int_params(names, params):
    for int_type in names:
        raw_val = params[int_type]
        if is_number(raw_val):
            params[int_type] = int(raw_val)
    return params
def SHAP(estimator,X,headers,antibiotic,fold):
    X=pandas.DataFrame(X,columns=headers)
    X=X.astype(float)
    explainer=shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X,check_additivity=False)
    shap.summary_plot(shap_values[0], X, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.subplots_adjust(left=0.2, top=0.95)
    plt.savefig(output_file_name+"/"+antibiotic+"/shap_value_bar_%s.svg"%fold)
    plt.savefig(output_file_name+"/"+antibiotic+"/shap_value_bar_%s.png"%fold,dpi=400)
    plt.close()
    for i in [0,1]:
        shap.summary_plot(shap_values[i], X,show=False,max_display=10,alpha=0.5)
        plt.subplots_adjust(left=0.2, top=0.95,bottom=0.2)
        plt.yticks(fontsize='medium')
        plt.xticks(fontsize='medium')
        plt.savefig(output_file_name+"/"+antibiotic+"/shap_value_top10_class%s_%s.svg"%(i,fold))
        plt.savefig(output_file_name+"/"+antibiotic+"/shap_value_top10_class%s_%s.png"%(i,fold),dpi=400)
        plt.close()    
    values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values[0]),axis=0),'features':headers})
    values.to_csv(output_file_name+"/"+antibiotic+"/shap_value_mean_class0_%s.csv"%fold,index=False,sep='\t')
###check parsed params
print("\n\n",time.asctime())
print(sys.argv[0])
print(args)
random.seed(random_seed)
open(output_file_name + '/log.txt','a').write(time.asctime()+"\n")
open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)  
###
###read metadata file
metadata=pandas.read_csv(metadata,sep='\t',dtype={'genome_id':str})
print('input',metadata.shape,len(list(set(metadata['genome_id']))))
metadata=metadata.drop_duplicates(subset=['genome_id','antibiotic'], keep="first")
metadata=metadata[metadata['nr_contig']<=300]
#read CSV files
if input_type=='pres':
    kmer=pandas.read_csv(data_file,sep='\t',index_col=0)
    kmer=kmer.T
    kmer['genome_id']=kmer.index
elif input_type=='vcf':
    kmer=pandas.read_csv(data_file,sep='\t',skiprows=3, usecols = list(set(metadata['genome_id'])), low_memory = True)
    open(output_file_name + '/log.txt','a').write("The number of SNP sites: %s\n"%kmer.shape[0]) 
    open(output_file_name + '/log.txt','a').write("The number of genomes: %s\n"%kmer.shape[1])  
    kmer=kmer.T
    # kmer['genome_id']=kmer.index
    genome_ids=list(kmer.index)
    headers=kmer.columns.values.tolist()
    kmer=pandas.DataFrame(data=np.c_[genome_ids,kmer],columns=['genome_id']+headers)
    open(output_file_name + '/log.txt','a').write("The number of SNP sites after selector: %s\n"%len(headers)) 
    snps=pandas.read_csv(data_file,sep='\t',skiprows=3, usecols = ['POS','REF','ALT'], low_memory = True)
    snp_rename=dict()
    for i in snps.index:
        snp_rename.update({i:str(int(snps['POS'][i]))+"_"+snps['REF'][i]+"_"+snps['ALT'][i]})
    kmer=kmer.rename(columns=snp_rename)
    print('SNPs',len(headers))
elif input_type=='pres+vcf':
    data_file=data_file.split(',')
    if len(data_file)!=2:
        print("Please input the pres and vcf tables in the form of 'pres_table,vcf_table'")
        print("Abort.")
        sys.exit()
    pres=pandas.read_csv(data_file[0],sep='\t',index_col=0)
    pres=pres.T
    
    kmer=pandas.read_csv(data_file[1],sep='\t',skiprows=3, usecols = list(set(metadata['genome_id'])), low_memory = True)
    open(output_file_name + '/log.txt','a').write("The number of SNP sites: %s\n"%kmer.shape[0]) 
    open(output_file_name + '/log.txt','a').write("The number of genomes: %s\n"%kmer.shape[1])  
    kmer=kmer.T
    genome_ids=list(kmer.index)
    headers=kmer.columns.values.tolist()
    kmer=pandas.DataFrame(data=np.c_[genome_ids,kmer],columns=['genome_id']+headers)
    overlap_genomes=[i for i in list(kmer['genome_id']) if i in list(pres.index)]
    print(len(overlap_genomes))
    open(output_file_name + '/log.txt','a').write("The number of overlapping genomes: %s\n"%len(overlap_genomes)) 
    kmer=kmer[kmer['genome_id'].isin(overlap_genomes)]
    pres=pres.loc[overlap_genomes]
    pres.reindex(list(kmer['genome_id'])) 
    kmer=pandas.DataFrame(data=np.c_[kmer,pres],columns=['genome_id']+headers+pres.columns.values.tolist())
    
    open(output_file_name + '/log.txt','a').write("The number of accessory genes: %s\n"%len(pres.columns.values.tolist())) 
    snps=pandas.read_csv(data_file[1],sep='\t',skiprows=3, usecols = ['POS','REF','ALT'], low_memory = True)
    snp_rename=dict()
    for i in snps.index:
        snp_rename.update({i:str(int(snps['POS'][i]))+"_"+snps['REF'][i]+"_"+snps['ALT'][i]})
    kmer=kmer.rename(columns=snp_rename)
    
    print('SNPs',len(headers))
    
elif input_type=='kmer':
    kmer=pandas.read_csv(data_file,sep='\t',dtype={'genome_id':str}) 
else:
    print('Unrecognized input type...')
    print('Abort')
    sys.exit()
kmers_cols=[i for i in kmer.columns if i not in ['genome_id','ST']]
kmer=kmer.fillna(0)

#select antibiotics to test
### this version includes a predefiined antibiotic list based on the overlapping samples in different phenotypes
print('anti',len(antis),antis)
open(output_file_name + '/log.txt','a').write("The number of antibiotics to be tested: %s\n"%len(antis))  
metadata=metadata[metadata['genome_id'].isin(list(split_file['genome_id']))]
kmer=kmer[kmer['genome_id'].isin(metadata['genome_id'])]

open(output_file_name + '/log.txt','a').write("Total training samples: %s\n"%split_file[split_file['split']!='val'].shape[0])  
open(output_file_name + '/log.txt','a').write("Total validation samples: %s\n"%split_file[split_file['split']=='val'].shape[0])  
metadata=metadata[metadata['antibiotic'].isin(antis)]
print('filtered',metadata.shape,len(list(set(metadata['genome_id']))))
open(output_file_name + '/log.txt','a').write("The row number of metadata after selecting antibiotics: {0}\n".format(metadata.shape[0]))  

### start to test on each antibiotic
evaluations=defaultdict(list)
for a in antis:
    os.mkdir(output_file_name+"/"+a)
    print("*****************")
    print(time.asctime(),"Start with\n",a)
    print()
    open(output_file_name + '/log.txt','a').write("\n**********\n{0}\n{1}:\n".format(time.asctime(),a))
    sub=metadata[metadata['antibiotic']==a]
    if estimator_type=='classifier' and len(set(sub[training_target]))>2: # only use R/S because gpboost doesn't support multiclass yet
        sub=sub[sub[training_target].isin(['R','S'])]
    open(output_file_name + '/log.txt','a').write("Number of samples: {0}\n".format(sub.shape[0]))
    print("Number of rows in metadata after filtering with antibiotic: {0}".format(sub.shape[0]))  ## should be the same as all in the new setup
    kmer_sub=kmer[kmer['genome_id'].isin(list(sub['genome_id']))]
    
    # sort metadata dataframe 
    sub=sub.set_index('genome_id')
    sub=sub.loc[kmer_sub['genome_id']]
    sub['genome_id']=list(sub.index)
    
    # training target
    label_encoder = LabelEncoder()
    y=np.array(label_encoder.fit_transform(sub[training_target]),dtype=int)
    open(output_file_name + '/log.txt','a').write("Label encoding SIR: %s \n" % label_encoder.classes_)
    X_strain=kmer_sub[kmers_cols]
    
    ###Create a dataframe for spliting (including genome_id and the phenotype (target))
    headers=X_strain.columns.values.tolist()
    open(output_file_name + '/log.txt','a').write("Header length: {0}\n\n".format(len(headers)))
    print('header',len(headers))
    X_df=pandas.DataFrame(data=np.c_[X_strain,y,sub.index],columns=headers+[training_target,'genome_id'])
    X_df = X_df.loc[:,~X_df.columns.duplicated()].copy()
     ##split into validation and training dataframes
    training=X_df[X_df['genome_id'].isin(list(split_file[split_file['split']!='val']['genome_id']))]
    validation=X_df[X_df['genome_id'].isin(list(split_file[split_file['split']=='val']['genome_id']))]
    if len(set(training[training_target]))==1:
        print("Only class %s in training set, skipping..."%label_encoder.classes_[list(set(training[training_target]))[0]])
        open(output_file_name + '/log.txt','a').write("Only class %s in training set, skipping...\n\n"%label_encoder.classes_[list(set(training[training_target]))[0]])
        continue
    open(output_file_name + '/log.txt','a').write("Number of samples for training: {0}\n".format(training.shape[0])) 
    
    ### create all the training test splits
    train_test_split_cols=list()
    clade_split_ratio_fold=defaultdict(list)
    clades=[i for i in list(set(split_file['split'])) if 'clade' in i or 'training' in i]
    open(output_file_name + '/log.txt','a').write("The number of test clades: {0}\n".format(len(clades))) 
    ##only include clades with both R and S phenotype
    included_clade=list()
    for clade in clades:
        clade_df=training[training['genome_id'].isin(list(split_file[split_file['split']==clade]['genome_id']))]
        if len(set(clade_df[training_target]))!=1:
            if clade_df[clade_df[training_target]==0].shape[0]>=50 and clade_df[clade_df[training_target]==1].shape[0]>=50 :
                included_clade.append(clade)
    included_clade.sort()
    if len(included_clade)==0:
        open(output_file_name + '/log.txt','a').write("No test clade included. Abort.") 
        continue
    
    open(output_file_name + '/log.txt','a').write("Included clades: {0}\n".format(included_clade)) 
    
    '''
    train on each clade 
    '''
    for clade in included_clade:
        open(output_file_name + '/log.txt','a').write("Clade: {0}\n".format(clade)) 
        X_clade=training[training["genome_id"].isin(list(split_file[split_file['split']==clade]['genome_id']))]
        open(output_file_name + '/log.txt','a').write("Sample size: {0} (R:{1},S:{2})\n".format(X_clade.shape[0],X_clade[X_clade[training_target]==0].shape[0],X_clade[X_clade[training_target]==1].shape[0])) 
            
        ##hyperparameter tuning split id
        split_ids=list(X_clade['genome_id'])
        ###preprocessors
        selector = VarianceThreshold()
        # scaler=StandardScaler()
        #hyperparameter tuning
        if params_output!=None and os.path.isfile(params_output+"/%s_%s_params.pkl"%(a,clade)):
            params=pickle.load(open(params_output+"/%s_%s_params.pkl"%(a,clade),'rb'))
        else:
            space = {'max_bin': hyperopt.hp.choice('max_bin', np.arange(50, 500, 50)),
                     'bagging_fraction':hp.uniform('bagging_fraction', 0.01, 1.0),
                     'bagging_freq':hp.uniform('bagging_freq', 0, 10.0),
                     'feature_fraction':hp.uniform('feature_fraction', 0.01, 1.0),
                     'subsample_for_bin':hp.uniform('subsample_for_bin', 30, int(0.8*X_df.shape[0])),
                     'max_depth': hp.quniform('max_depth', 1, 16, 1),
                      'learning_rate':hp.uniform('learning_rate', 0.01, 1.0),
                      'lambda_l2':hp.uniform('lambda_l2', 0.0, 100.0),
                        'min_data_in_leaf': hyperopt.hp.choice('min_data_in_leaf', np.arange(1, 300)),
                        'min_gain_to_split': hyperopt.hp.choice('min_gain_to_split', np.arange(0, 15)),
                        'num_leaves': hyperopt.hp.choice('num_leaves', np.arange(2, 100, 1))}
            def objective_sklearn(params):
                int_types=['max_bin','max_depth','bagging_freq','num_leaves','min_data_in_leaf','subsample_for_bin']
                params = convert_int_params(int_types, params)
                params.update({'verbose':-1,'objective': objective,'boosting':'gbdt','linear_tree':False,
                                  'tree_learner':'serial','num_threads':0,'seed':np.random.seed(random_seed),'deterministic':True,
                                  'force_col_wise':True,'bagging_seed':np.random.seed(random_seed),'feature_fraction_seed':np.random.seed(random_seed),'extra_trees':True})
                #get the mean score of 5 folds
                kf=sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=np.random.seed(random_seed))
                scores=list()
                for train_index, test_index in kf.split(split_ids):##split the combined training set into train and test based on guideid
                    train_index=np.array(split_ids)[train_index]
                    test_index=np.array(split_ids)[test_index]
                    # print(train_index,test_index)
                    X_train = X_clade[X_clade['genome_id'].isin(train_index)]
                    y_train=np.array(X_train[training_target],dtype=int)
                    X_train=X_train[headers]
                
                    X_train=selector.fit_transform(X_train)
                    # X_train=scaler.fit_transform(X_train)
                    mask=selector.get_support()
                    if False not in mask:
                        new_headers=headers
                    else:
                        if len(mask)==len(headers):
                            new_headers=[]
                            for i in range(len(mask)):
                                if mask[i]:
                                    new_headers.append(headers[i])
                    X_train=pandas.DataFrame(data=X_train,columns=new_headers)
                    X_test = X_clade[X_clade['genome_id'].isin(test_index)]
                    y_test=np.array(X_test[training_target],dtype=int)
                    X_test=X_test[new_headers]
                    if len(set(y_test))==1: ##only one class in train/test
                        continue
                    train_data = lgb.Dataset(np.array(X_train,dtype=float), label=y_train)
                    estimator = lgb.train(params=params, train_set=train_data, num_boost_round=100)
                    if estimator_type=='classifier':
                        predictions_proba=estimator.predict(np.array(X_test,dtype=float))
                        # print(predictions_proba)
                        if objective!='multiclass':
                            predictions=[1  if i>=0.5 else 0 for i in predictions_proba]
                        else:
                            predictions=list()
                            for i in predictions_proba:
                                predictions.append(np.argmax(i))
                        scores.append(-1*sklearn.metrics.balanced_accuracy_score(y_test, predictions))
                    elif estimator_type=='regressor':
                        predictions=estimator.predict(X_test)
                        scores.append(sklearn.metrics.mean_absolute_error(y_test, predictions))
                score=np.mean(scores)
                #using logloss here for the loss but uncommenting line below calculates it from average accuracy
                result = {"loss": score, "params": params, 'status': hyperopt.STATUS_OK}
                return result
            
            
            n_trials = 20
            trials = Trials()
            try:
                best = fmin(fn=objective_sklearn,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=n_trials,
                        trials=trials,
                        early_stop_fn=no_progress_loss(5),
                        rstate=np.random.default_rng(random_seed))
            except hyperopt.exceptions.AllTrialsFailed:
                n_trials = 100
                trials = Trials()
                best = fmin(fn=objective_sklearn,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=n_trials,
                        trials=trials,
                        early_stop_fn=no_progress_loss(5),
                        rstate=np.random.default_rng(random_seed))
            except ValueError:
                continue
            idx = np.argmin(trials.losses())
            params = trials.trials[idx]["result"]["params"]
    
        params.update({'objective': objective,'boosting':'gbdt','linear_tree':False,'verbose':-1,
                              'tree_learner':'serial','num_threads':0,'seed':np.random.seed(random_seed),'deterministic':True,
                              'force_col_wise':True,'bagging_seed':np.random.seed(random_seed),'feature_fraction_seed':np.random.seed(random_seed),'extra_trees':True})
        print(params)
        
        if params_output!=None and os.path.isfile(params_output+"/"+a+"_"+clade+"_params.pkl")==False:
            pickle.dump(params, open(params_output+"/%s_%s_params.pkl"%(a,clade), 'wb'))
        open(output_file_name + '/log.txt','a').write("Params: {0}\n".format(params)) 
    
        
        X_selected=selector.fit_transform(X_clade[headers])
        mask=selector.get_support()
        if False not in mask:
            preprocessed_headers=headers
        else:
            preprocessed_headers=[]
            for i in range(len(mask)):
                if mask[i]==True:
                    preprocessed_headers.append(headers[i])
        X_selected=pandas.DataFrame(data=X_selected,columns=preprocessed_headers)
        train_data = lgb.Dataset(np.array(X_selected,dtype=float), label=np.array(X_clade[training_target],dtype=float))
        estimator = lgb.train(params=params, train_set=train_data, num_boost_round=100)
        
        estimator.save_model(output_file_name+"/"+a+'/bst_model_%s.json'%clade)
        SHAP(estimator,X_selected,preprocessed_headers,a,clade)
        open(output_file_name + '/log.txt','a').write("\nParams for filnal saved model: {0}\n\n".format(estimator.params)) 
 

 
open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))    

print(time.asctime(),"Done.")
