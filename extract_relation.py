import numpy as np
import pandas as pd
from tqdm import tqdm

from rulebased import *
from french_patterns import *
from corpushelpers import postags

from joblib import Parallel,delayed

yes_questions = ["QUXVlc3Rpb246MTQ4","QUXVlc3Rpb246MTQ2","QUXVlc3Rpb246MTU0","QUXVlc3Rpb246MTUy","travaux d'isolation|commun|isolation|ans|prix|chauffage"]



def prep_4_writing(buffer, question_id,batch_number):
    fin_res=[]
    for _, row in buffer.iterrows():
        if row.rel:
            for r in row.rel:
                fin_res.append([row.question, row.contribution, *r, row.Start.strftime("%d/%m/%Y"),row.End.strftime("%d/%m/%Y")])
    df = pd.DataFrame(fin_res,columns="Question Contribution Source Target Type TextExtract Start End".split())
    df.to_csv("{0}_{1}.csv".format(question_id,batch_number))

def work(data,transf):
    pos_tag_fin = pip_FR.pipe(data)
    if transf:
        pos_tag_fin = tran_pip.pipe(pos_tag_fin)
    else:
        pos_tag_fin = eng_pip.pipe(pos_tag_fin)
    res = pip_rule.pipe(pos_tag_fin)
    return res

# READ INPUT
print("Loading Dataset")
data = pd.read_csv("../results_cp_dt.csv",index_col=0)
data = data.drop_duplicates(subset=["question","contribution"])
data = data[~(data.question.isin(yes_questions))]
print("Data Loaded !")
data= data.fillna("")


n_split = 5
transf=True
enga=False
constat=False

#data = pd.read_csv("./sample2.csv",index_col=0)

# Extract KEYWORDS
kw = data.keywords.unique()
kw = [str(x).split("|") for x in kw]
kw = np.unique(np.hstack(kw))
kw = [i.split() for i in kw if i]


# ----------------------------------------
# ---------- RULES DEFINITIONS -----------
# ----------------------------------------
#BASIC parsing
pip_FR = PipelineParser()
pip_FR.rules.append(ParsingRule(np.asarray(kw),"KW",0))
pip_FR.rules.append(ParsingRule(np.asarray(kw),"KW",1))
pip_FR.rules.append(PruningRule(np.asarray(tags_to_keep),1))
pip_FR.rules.append(MergeRule(tag_to_merge="KW"))
pip_FR.rules.append(PruningRule(np.asarray([["PUN"]]),1,False))
pip_FR.rules.append(ParsingRule(np.asarray(verbs_POS_list),"VER",1))
pip_FR.rules.append(ParsingRule(np.asarray(dets_list),"DET",1))
pip_FR.rules.append(ParsingRule([["être"]],"ETRE",2))


# RELATION EXTRACTOR
pip_rule = RelationIdentificationPipeline()

# TRANSFORMATION VERBS PIPELINE IDENTIFICATION
tran_pip = PipelineParser()
tran_pip.rules.append(ParsingRule(verbs_selected["TRAN"],"TRAN",2))
# Engagment VERBS PIPELINE IDENTIFICATION
eng_pip = PipelineParser()
eng_pip.rules.append(ParsingRule(verbs_selected["ENGA"],"ENGA",2))
# Je
je_pip = PipelineParser()
je_pip.rules.append(ParsingRule([["je"]],"JE",2))



if transf:
    pip_rule.rules.append(RelationRule([["TRAN","DET","KW"]],"changement",0,2,2,1))
    pip_rule.rules.append(RelationRule([["ADV","TRAN","ADV","DET","KW"]],"nepas_changment",1,4,2,1))
elif enga:
    pip_rule.rules.append(RelationRule([["ENGA","DET","KW"]],"engagement",0,2,2,1))
    pip_rule.rules.append(RelationRule([["ADV","ENGA","ADV","DET","KW"]],"nepas_enga",1,4,2,1))
elif constat:
    #patterns,rule_name,src_position,tar_postion,value_idx=0,pattern_idx=0):
    pip_rule.rules.append(RelationRule([["KW","ETRE","ADJ"]],"constat",0,2,2,1))

# Run Relation Extraction 
import gc, os
for name, group in data.groupby("question"):
    print("Processing question : ",name)
    splited = np.array_split(group,n_split)
    ix=0
    while splited:
        print("Working on Batch",ix)
        data_ix = splited.pop(0)
        # if os.path.exists("{0}_{1}.csv".format(name,ix)):
        #     ix+=1
        #     continue
        print("PosTagging in Progress")
        data_ix = postags(data_ix,text_column="reponse")
        print("Extract Relations in Batch",ix)
        buffer = Parallel(n_jobs=12)(delayed(work)(pos_tag,transf) for ij,pos_tag in tqdm(enumerate(data_ix.pos_tag.values),total=len(data_ix)))

        data_ix['Start'] = pd.to_datetime(data_ix.publishedat).dt.to_period('D')
        data_ix["End"] = data_ix.Start.apply(lambda x: x+1)
        data_ix["rel"] = buffer
        print("Save Data")
        prep_4_writing(data_ix,name,ix)
        
        print("Data saved")
        gc.collect()
        print("Buffer empty")
        ix+=1
    break


