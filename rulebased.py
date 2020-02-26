import numpy as np
import pandas as pd
import warnings

def match_sequences(seqs,dataset):
    """
    Return matched sequence start,end positions from a dataset
    
    Parameters
    ----------
    seq : list
        sequence
    dataset : list
        dataset
    
    """
    N=len(seqs)
    if N < 1:
        warnings.warn("Sequence Empty")
        return []

    if isinstance(dataset,list):
        dataset=np.asarray(dataset)
    if isinstance(seqs,list):
        seqs=np.array(seqs)
        
    prefixes_dict=np.array([[seq[0],(seq,i,len(seq))] for i,seq in enumerate(seqs)])
    prefixes=list(prefixes_dict[:,0])
    prefix_ind=np.where(np.isin(dataset,prefixes))[0]
    results=[]
    for idx in prefix_ind:
        cond = np.where(prefixes_dict[:,0] == dataset[idx])
        for el in prefixes_dict[cond]:
            start,end=idx,idx+el[1][-1]
            try:
                if (dataset[start:end].tolist() == el[1][0]):
                    results.append([el[1][1],start,end])
            except ValueError:
                if (dataset[start:end].tolist() == el[1][0].tolist()):
                    results.append([el[1][1],start,end])
    return results

class Rule(object):
    def __init__(self):
        pass
    def parse_tags(self,pos_tags):
        """
        add tag to certains sequence
        
        Parameters
        ----------
        pos_tags : 2D array (token,tag,lemma)
            Post-tags array returned by TreeTagger
        sequences : 2D array [["tok1","tok2"],["tok3","tok5"]]
            patterns to match
        new_tag : string
            new tag
        """
        raise NotImplementedError("Rule is an abstract class")

#----------------------------------------------------------------------------------------------------
# PARSING RULES
#----------------------------------------------------------------------------------------------------

class ParsingRule(Rule):
    """
    ParsingRule is used to replace par-of-speech tag(s) of tokens that match certain patterns by a new tag. 
    """
    def __init__(self,patterns,new_tag,pattern_idx=0):
        """
        Constructor of ParsingRule
        
        Parameters
        ----------
        patterns : 2D array [[pat1],[pat2]]
            patterns to match
        new_tag : str
            tag to replace
        pattern_idx : int, optional
            columns in the part-of-speech output used for searching given patterns
        
        Raises
        ------
        ValueError
            If pattern_idx not in [0,1,2]
        """
        Rule.__init__(self)
        if not isinstance(pattern_idx,int):
            if not pattern_idx in [0,1,2]:
                raise ValueError("Pattern Index should be between 0 and 2.")
        self.patterns = patterns
        self.new_tag = new_tag
        self.pattern_idx = pattern_idx

    def parse_tags(self,pos_tags):
        
        tags = np.asarray(pos_tags).copy()
        try:
            ind_seq = np.asarray(match_sequences(dataset=tags[:,self.pattern_idx],seqs=self.patterns))[:,1:3]
            index_seq = [np.arange(index[0],index[1]).tolist() for index in ind_seq]
            for idx in index_seq:
                tags[idx,1]= self.new_tag
                
        except IndexError as e:
            pass # If no tokens that match the pattern found
        
        return tags

class PruningRule(Rule):
    """
    PruningRule is used to prune tokens that match given patterns. It can be used differently, either the patterns are used to detect tokens that must
    be deleted or kept.
    
    """
    def __init__(self,patterns,pattern_idx=0,keep_only=True):
        """
        PruningRule constructor
        
        Parameters
        ----------
        patterns : 2D array [[pat1],[pat2]]
            patterns to match
        pattern_idx : int, optional
            columns in the part-of-speech output used for searching given patterns
        keep_only : bool, optional
            tokens detected in a pattern are kept if true, or deleted if false, by default True
        
        Raises
        ------
        ValueError
            If pattern_idx not in [0,1,2]
        """
        Rule.__init__(self)
        if not isinstance(pattern_idx,int):
            if not pattern_idx in [0,1,2]:
                raise ValueError("Pattern Index should be between 0 and 2.")

        self.patterns = patterns
        self.pattern_idx = pattern_idx


        self.keep_only = keep_only
    def parse_tags(self,pos_tags):

        tags = np.asarray(pos_tags).copy()
        try:
            indxs = np.asarray(match_sequences(dataset=tags[:,self.pattern_idx],seqs=self.patterns))[:,1]
            if self.keep_only:
                tags = tags[indxs]
            else:
                tags = np.delete(tags, indxs,axis=0)
        except IndexError as e:
            pass

        return tags

class MergeRule(Rule):
    def __init__(self,tag_to_merge):
        Rule.__init__(self)
        self.tag_to_merge = tag_to_merge
    
    def parse_tags(self, pos_tags):
        return self.merge_kw(pos_tags)

    def merge_set(self,pos_tags):
        """
        Return set of tagged n-gram
        
        Parameters
        ----------
        pos_tags : 2D array
            POS of a doc
        
        Returns
        -------
        2D-Array
            subset-indices of `pos_tags` to merge
        """
        try:
            tag_pos = np.where(pos_tags[:,1] == self.tag_to_merge)[0]
        except IndexError:
            return []
        # IF NO TOKENS ASSOCIATED to specified TAG
        if not len(tag_pos)>0:
            return []
        # Distance between position of each token found
        diff = np.append(tag_pos[1:],tag_pos[-1]) - tag_pos

        set_of_tagged_tokens,current_set = [],[]
        for ix,d in enumerate(diff):
            current_set.append(tag_pos[ix])
            if d>1: # if distance >1 --> moving to a new set of tokens
                set_of_tagged_tokens.append(current_set)
                current_set=[]

        # If loop over and curr not empty
        if current_set:set_of_tagged_tokens.append(current_set)

        return set_of_tagged_tokens
    
    def merge_kw(self,pos_tags):

        tags2 = pos_tags.copy()
        tags2 =tags2.astype(object)
        # Found SET of TOKENS to merge
        ix_merge = self.merge_set(tags2)
        if not ix_merge:
            return tags2
        not_included=[]
        for m_ixs in ix_merge:
            if len(m_ixs)>1:
                tags2[m_ixs[0],0]=" ".join(tags2[m_ixs,0])
                tags2[m_ixs[0],2]=" ".join(tags2[m_ixs,2])
                not_included.extend(m_ixs[1:])
        
        return np.delete(tags2,not_included,axis=0)

class PipelineParser:
    def __init__(self):
        self.__rules = []
    
    def pipe(self,pos_tags):
        tags = pos_tags.copy()
        for rule in self.rules:
            tags = rule.parse_tags(tags)
        return tags

    @property
    def rules(self): 
        return self.__rules 

    @rules.setter
    def rules(self,rule):
        self.__rules.append(rule)


#----------------------------------------------------------------------------------------------------
#RELATIONSHIP IDENTIFICATION RULES
#----------------------------------------------------------------------------------------------------
def is_whitespace_before(token_p,token_n):
    if token_n in set(".,-)") or token_n[0] in set("-"):
        return False
    if token_p[-1] in set("'\"\’-("):
        return False
    return True

def get_white_space(tagged_text):
    try: # if numpy array
        tagged_text = tagged_text.tolist()
    except:
        pass
    tagged_text[0].append(tagged_text[0][0])
    for ix in range(1,len(tagged_text)):
        token_p=tagged_text[ix-1][0]
        token_n = tagged_text[ix][0]
        if is_whitespace_before(token_p,token_n):
            tagged_text[ix].append(" "+tagged_text[ix][0])
        else:
            tagged_text[ix].append( tagged_text[ix][0])
    return np.array(tagged_text)

class RelationRule(Rule):
    def __init__(self,patterns,rule_name,src_position,tar_postion,value_idx=0,pattern_idx=0):
        Rule.__init__(self)
        self.patterns = patterns
        self.src_position,self.tar_postion = src_position,tar_postion
        self.pattern_idx = pattern_idx
        self.rule_name=rule_name
        self.value_idx = value_idx
    def parse_tags(self, pos_tags):
        results = []
        
        try:
            pos_tags = get_white_space(pos_tags) # For text extract
            indxs = np.asarray(match_sequences(dataset=pos_tags[:,self.pattern_idx],seqs=self.patterns))[:,1:3]
            for idx in indxs:
                src = pos_tags[idx[0]+self.src_position,self.value_idx]
                tar = pos_tags[idx[0]+self.tar_postion,self.value_idx]
                text = "".join(pos_tags[idx[0]+self.src_position:idx[0]+self.tar_postion+1,-1])
                results.append([src,tar,self.rule_name,text])
        except IndexError as e:
            pass
        return results
        
        

class RelationIdentificationPipeline:
    def __init__(self):
        self.__rules = []
    
    def pipe(self,pos_tags):
        relation_occurence_found = []
        for r in self.rules:
            relation_occurence_found.extend(r.parse_tags(pos_tags))
        return relation_occurence_found #pd.DataFrame(relation_occurence_found,columns="src tar type".split()) 

    @property
    def rules(self): 
        return self.__rules 

    @rules.setter
    def rules(self,rule):
        if not isinstance(RelationRule):
            raise TypeError
        self.__rules.append(rule)


if __name__ == "__main__":
    from lib.treetagger import TreeTagger

    tt = TreeTagger(language="french")

    pos_tags = tt.tag("""Les perspectives économiques mondiales s’assombrissent, sur fond de tensions commerciales et géopolitiques. L’Organisation de coopération et de développement économiques (OCDE) a abaissé, jeudi 19 septembre, ses prévisions de croissance mondiale de 0,3 et 0,4 point de PIB pour 2019 et 2020, par rapport à celles de mai. Si, comme le prévoit désormais l’OCDE, la croissance atteint 2,9 % en 2019 et 3 % l’année suivante, l’économie mondiale enregistrerait ses pires performances depuis la crise financière de 2008. Un freinage qui touche à la fois les pays riches et les pays émergents. « Il y a un risque de ralentissement structurel », souligne Laurence Boone, chef économiste de l’OCDE, qui cite l’impasse des négociations commerciales entre la Chine et les Etats-Unis, la menace du Brexit ou encore le regain de tensions entre le Japon et la Corée du Sud.""")

    pip = PipelineParser()
    pip.rules.append(PruningRule(patterns=[["VER:pres"]],pattern_idx=1,keep_only=False))
    print(pos_tags)
    print("-----")
    print(pip.pipe(pos_tags))