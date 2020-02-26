"""
French TreeTagger Part-of-Speech Tags

Achim Stein, April 2003
------------
ABR	abreviation
ADJ	adjective
ADV	adverb
DET:ART	article
DET:POS	possessive pronoun (ma, ta, ...)
INT	interjection
KON	conjunction
NAM	proper name
NOM	noun
NUM	numeral
PRO	pronoun
PRO:DEM	demonstrative pronoun
PRO:IND	indefinite pronoun
PRO:PER	personal pronoun
PRO:POS	possessive pronoun (mien, tien, ...)
PRO:REL	relative pronoun
PRP	preposition
PRP:det	preposition plus article (au,du,aux,des)
PUN	punctuation
PUN:cit	punctuation citation
SENT	sentence tag
SYM	symbol
VER:cond	verb conditional
VER:futu	verb futur
VER:impe	verb imperative
VER:impf	verb imperfect
VER:infi	verb infinitive
VER:pper	verb past participle
VER:ppre	verb present participle
VER:pres	verb present
VER:simp	verb simple past
VER:subi	verb subjunctive imperfect
VER:subp	verb subjunctive present
"""


# Verbe de transformation
transf_verb = "accélérer, ralentir, limiter, construire, détruire, baisser, hausser, changer, éviter, obliger, interdire, baisser, autoriser, développer, réduire, augmenter, sortir, modifier, supprimer, permettre, cesser, imposer, adapter, arrêter, démarrer, sortir, cesser, installer, créer".split(", ")

# Verbe d'engagement
engag_verb = "préconiser, former, assurer, investir, entretenir, choisir, investir, contrôler, convaincre, obliger, agir, limiter, mobiliser, informer, appliquer, admettre, encourager, sanctionner, limiter, favoriser, proposer, aider, réguler, subventionner, privilégier, rejeter, permettre, oser, autoriser, taxer".split(", ")

# Tags for selected verb
verbs_selected = {
    "TRAN":[[i]for i in transf_verb],
    "ENGA":[[i]for i in engag_verb]
}

verbs_POS_list = [["VER:futu"],
["VER:impe"],
["VER:impf"],
["VER:infi"],
["VER:pper"],
["VER:ppre"],
["VER:pres"],
["VER:simp"],
["VER:subi"],
["VER:subp"],["VER:cond"]]

dets_list = [["DET:ART"],
["DET:POS"]]

tags_to_keep = [["NOM"],["PRP"],["PRO:PER"],["PRO"],["ADJ"],["ADV"],["KW"],["PUN"]]
tags_to_keep.extend(verbs_POS_list)
tags_to_keep.extend(dets_list)