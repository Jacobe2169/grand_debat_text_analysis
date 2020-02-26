"""
Helpers
"""

from treetagger import TreeTagger
import numpy as np




def parse_output(x):
    if "<unknown>" in x[:,2]:
        idx = np.where(x[:,2]=="<unknown>")
        x[idx,2]=x[idx,0]
    return x

def postags(data,text_column="reponse",lang="french"):
    """
    Return TreeTagger Part-Of-Speech outputs for large corpus. 
    
    Parameters
    ----------
    texts : list of str
        corpus
    lang : str, optional
        {french, spanish, english, ...}, by default "french"
    nb_split : int, optional
        number of text send to TreeTagger at each loop, by default 50
    """
    tree_tagger = TreeTagger(language=lang)
    res = tree_tagger.tag("\n##############END\n".join(data[text_column].values))
    res = [r for r in res if len(r) ==3 ]
    res = np.asarray(res)
    indexes = np.where(res[:,0]=="##############END")[0]
    pos_tag_data  = np.asarray([parse_output(i[1:]) for i in np.split(res,indexes)])
    data["pos_tag"] = pos_tag_data
    return data
