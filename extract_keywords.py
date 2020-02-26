# Core Librairies
import json

# Data
import pandas as pd
import numpy as np

# NLP
import spacy
from biotex import BiotexWrapper
from stop_words import get_stop_words
fr_stop = get_stop_words("french")

#ESTETHICS
from tqdm import tqdm

# PARALLEL
from joblib import Parallel,delayed

############################################################################
#                        NLP FUNCTION
############################################################################


def extract_and_treat_keywords_terminology(texts):
    """
    Extract a normalized terminology from given texts.
    
    Parameters
    ----------
    texts : list of str
        List of documents
    
    Returns
    -------
    pd.DataFrame
        keywords terminology
    """
    # First, we use Biotex to build the first version of the keywords terminology
    biot = BiotexWrapper(language="french")

    terminology = biot.terminology(corpus=texts)
    terminology["gram"] = terminology.term.apply(lambda x: len(x.split()))

    return terminology


def if_in(keywords,text):
    """
    Return keywords that appears in 
    
    Parameters
    ----------
    keywords : list of str
        keywords
    text : str
        text
    
    Returns
    -------
    str
        keywords list separated by a pipe
    """
    result_=set([])
    for word in keywords:
        if word in text:
            result_.add(word)
    return "|".join(list(result_))



############################################################################
#                        MAIN CODE
############################################################################
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("input_data")
parser.add_argument("question_data")

args = parser.parse_args("./LA_TRANSITION_ECOLOGIQUE.csv ./questionsTRANSITION.json".split())

df = pd.read_csv(args.input_data,dtype={"authorZipCode":str})
df = df.fillna("")

data_questions = json.load(open(args.question_data))

df.rename(columns=data_questions["all_question"],inplace=True)

# EXTRACT and SAVE terminology extracted for each question
for i in tqdm(range(1,len(data_questions["all_question"])+1)):
    termi_i = extract_and_treat_keywords_terminology(df[i].values)
    termi_i.to_csv("./terminologies_extracted/question_{0}.csv".format(i))

for i in tqdm(range(1,len(data_questions["all_question"])+1)):
    term_i = pd.read_csv("./terminologies_extracted/question_{0}.csv".format(i))
    kw = term_i.term.values.tolist()
    kw = [str(k) for k in kw]
    kw = set([k for k in kw if len(k)>2 and (k not in fr_stop)])
    df["{0}_kw".format(i)]=Parallel(n_jobs=-1,backend="multiprocessing")(delayed(if_in)(kw,x) for x in tqdm(df[i].values))

# Extract Location from response for each question
N_questions = len(data_questions["all_question"])

### SAVE Extraction only
df.to_csv("{0}_with_keywords.csv".format(args.input_data.replace(".csv","")))
