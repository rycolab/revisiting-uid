#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()


# In[ ]:


import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict


# In[ ]:


get_ipython().run_cell_magic('R', '', "install.packages('lme4')\ninstall.packages('MuMIn')\ninstall.packages('lmerTest')\ninstall.packages('ggplot2')")


# ## Preprocessing

# ### LM Scoring

# In[ ]:


STRIDE = 200
def score_gpt(sentence):
      with torch.no_grad():
        all_log_probs = torch.tensor([], device=model.device)
        offset_mapping = []
        start_ind = 0

        while True:
            encodings = tokenizer(sentence[start_ind:], max_length=1022, truncation=True, return_offsets_mapping=True)
            tensor_input = torch.tensor([[tokenizer.bos_token_id] + encodings['input_ids'] + [tokenizer.eos_token_id]], device=model.device)
            output = model(tensor_input, labels=tensor_input)
            shift_logits = output['logits'][..., :-1, :].contiguous()
            shift_labels = tensor_input[..., 1:].contiguous()
            log_probs = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
            assert torch.isclose(torch.exp(sum(log_probs)/len(log_probs)),torch.exp(output['loss']))
            offset = 0 if start_ind == 0 else STRIDE-1
            all_log_probs = torch.cat([all_log_probs,log_probs[offset:-1]])
            offset_mapping.extend([(i+start_ind, j+start_ind) for i,j in encodings['offset_mapping'][offset:]])
            if encodings['offset_mapping'][-1][1] + start_ind == len(sentence):
                break
            start_ind += encodings['offset_mapping'][-STRIDE][1]
        return np.asarray(all_log_probs.cpu()), offset_mapping


def score_bert(sentence):
    mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
    with torch.no_grad():
        all_log_probs = []
        offset_mapping = []
        start_ind = 0
        while True:
            encodings = tokenizer(sentence[start_ind:], max_length=512, truncation=True, return_offsets_mapping=True)
            tensor_input = torch.tensor([encodings['input_ids']], device=model.device)
            mask_input = tensor_input.clone()
            offset = 1 if start_ind == 0 else STRIDE
            for i, word in enumerate(encodings['input_ids'][:-1]):
                if i < offset:
                    continue
                mask_input[:,i]=mask_id
                output = model(mask_input, labels=tensor_input)
                log_probs = torch.nn.functional.log_softmax(output['logits'][:,i], dim=-1).squeeze(0)
                all_log_probs.append(-log_probs[tensor_input[0,i]].item())
                mask_input[:,i] = word
            
            offset_mapping.extend([(i+start_ind, j+start_ind) for i,j in encodings['offset_mapping'][offset:-1]])
            if encodings['offset_mapping'][-2][1] + start_ind >= (len(sentence)-1):
                break
            start_ind += encodings['offset_mapping'][-STRIDE-1][1]
            
        return all_log_probs, offset_mapping


# In[ ]:


MODEL = "gpt"

if MODEL == "bert":
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    score = score_bert
else:
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    score = score_gpt
model.eval()
model = model.cuda()


# In[ ]:


# Test for above function
a=['there is a books on the desk',
                'there is a plane on the desk',
                        'there is a book in the desk']
scores = [score(i) for i in a]
a, b = score("This is a test "*1)


# In[ ]:


import nltk
#nltk.download('punkt')
def string_join(x, j=''):
    return j.join(x)

def get_word_mapping(words):
    offsets = []
    pos = 0
    for w in words:
        offsets.append((pos,pos+len(w)))
        pos += len(w) + 1
    return offsets

def string_to_log_probs(string, probs, offsets):
    words = string.split()
    agg_log_probs = []
    word_mapping = get_word_mapping(words)
    cur_prob = 0
    cur_word_ind = 0
    for lp, ind in zip(probs, offsets):
        cur_prob += lp
        if ind[1] == word_mapping[cur_word_ind][1]:
            agg_log_probs.append(cur_prob)
            cur_prob = 0
            cur_word_ind += 1
    return agg_log_probs


# ### Unigram model

# In[ ]:


def get_ngrams(sentence, n=3):
    words = sentence.split()
    words = ["BOS"]*(n-1) + words + ["EOS"]
    ngrams = []
    for i in range(len(words)-n+1):
        ngrams.append((tuple(words[i:i+n-1]), words[i+n-1]))
    return ngrams

def normalize(root, log=True):
    for prefix in root.keys():
        counts = root[prefix]
        total_counts = np.sum([counts[word] for word in counts.keys()])
        if log:
            root[prefix] = {k: np.log(v)- np.log(total_counts) for (k,v) in counts.items()}
        else:
            root[prefix] = {k: v/total_counts  for (k,v) in counts.items()}
    return root

def sampling_format(root, normalize=True):
    for prefix in root.keys():
        v = root[prefix]
        words = list(v.keys())
        counts = np.array([v[word] for word in words])
        if normalize:
            counts = counts/np.sum(counts)
        root[prefix] = (words, np.log(counts))
    return root

def create_ngram_model(filename, n, outfile):
    with open(filename, 'r') as f:
        root = defaultdict(lambda: defaultdict(int))
        for sentence in f:
            if not sentence:
                continue
            ngrams = get_ngrams(sentence.lower().strip(), n)
            for ngram in ngrams:
                root[ngram[0]][ngram[1]] += 1
        root = normalize(root, log=False)
        pickle.dump(dict(root), open(outfile, "wb"))
        return root

n = 1
#root = create_ngram_model("wikitext-103/wiki.train.tokens", 1, "unigram.pkl")
root = pickle.load(open("unigram.pkl", "rb"))


# ### Corpus Statistics

# In[ ]:


from scipy.special import log_softmax
def ent(x):
    #sometimes given as surprisal and sometimes as log-prob
    l_soft = log_softmax(-abs(x))
    return -sum(np.exp(l_soft)*l_soft)
def norm_ent(x):
    return ent(x)/np.log(len(x))
def local_diff(x):
    d = 0
    for i in range(len(x)-1):
        d += abs(x[i+1]-x[i])
    return d
def local_diff2(x):
    d = 0
    for i in range(len(x)-1):
        d += (x[i+1]-x[i])**2
    return d

def corpus_stats(df, probs_file=''):
    stats = defaultdict(lambda: defaultdict(list))
    try:
        stats['log_probs'] = pickle.load(open(probs_file, "rb"))
    except:
        stats['log_probs'] = {}
    for i, s in df:
      # remove trailing white space
        s = s[:-1] if not s[-1] else s
        stats['sent_markers'][i] = np.cumsum([len(sen.split()) for sen in nltk.sent_tokenize(s)])
        stats['split_string'][i] = s.split()
        if i not in stats['log_probs']:
            stats['log_probs'][i] = score(s)
        stats['agg_log_probs'][i] = np.array(string_to_log_probs(s, stats['log_probs'][i][0],stats['log_probs'][i][1]))
        for j in range(len(stats['sent_markers'][i])):
            last = 0 if not j else stats['sent_markers'][i][j-1]
            stats['variances'][i].append(np.var(stats['agg_log_probs'][i][last:stats['sent_markers'][i][j]]))
            stats['maxes'][i].append(np.amax(stats['agg_log_probs'][i][last:stats['sent_markers'][i][j]]))        
            stats['means'][i].append(np.mean(stats['agg_log_probs'][i][last:stats['sent_markers'][i][j]]))
            stats['entropies'][i].append(ent(stats['agg_log_probs'][i][last:stats['sent_markers'][i][j]]))
            stats['local_diff'][i].append(local_diff(stats['agg_log_probs'][i][last:stats['sent_markers'][i][j]]))
            stats['local_diff2'][i].append(local_diff2(stats['agg_log_probs'][i][last:stats['sent_markers'][i][j]]))
    return stats

# pickle.dump(stats['log_probs'], open("gpt_probs.pkl", "wb"))


# In[ ]:


# Need all the function definitions for column names. o.w. there's no consistent mapping
def power(x, y): return np.sum(abs(x))**y
def power2(x): return power(x,2)
def power3(x): return power(x,3)
def power3_25(x): return power(x,3.25)
def power3_5(x): return power(x,3.5)
def power3_75(x): return power(x,3.75)
def power2_5(x): return power(x,2.5)
def power2_25(x): return power(x,2.25)
def power1_25(x): return power(x,1.25)
def power1_5(x): return power(x,1.5)
def power1_75(x): return power(x,1.75)
def power2_75(x): return power(x,2.75)

def produce_aggregate_per_sentence(main_df, stats):
    aggregate_per_sentence = main_df.groupby(by=["WorkerId","text_id", "sentence_num"]).agg({"time":[np.sum, np.mean, np.count_nonzero], 
                                                                                                               "log_prob": [np.sum, power1_25, power2_75, power3, power2, power1_5, power1_75, power2_25, power2_5,power3_25, power3_5,power3_75], 
                                                                                                               "word_len":[np.sum, np.mean], 
                                                                                                               "freq":[np.sum, np.mean]}).reset_index()
    aggregate_per_sentence.columns = ['_'.join(col).strip() for col in aggregate_per_sentence.columns.values]
    aggregate_per_sentence['log_prob_mean'] = aggregate_per_sentence.apply(lambda x: stats['means'][x['text_id_']][int(x['sentence_num_'])], axis=1)
    aggregate_per_sentence['log_prob_max'] = aggregate_per_sentence.apply(lambda x: stats['maxes'][x['text_id_']][int(x['sentence_num_'])], axis=1)
    aggregate_per_sentence['log_prob_ent'] = aggregate_per_sentence.apply(lambda x: stats['entropies'][x['text_id_']][int(x['sentence_num_'])], axis=1)
    aggregate_per_sentence['log_prob_var'] = aggregate_per_sentence.apply(lambda x: stats['variances'][x['text_id_']][int(x['sentence_num_'])], axis=1)
    aggregate_per_sentence['log_prob_ldiff'] = aggregate_per_sentence.apply(lambda x: stats['local_diff'][x['text_id_']][int(x['sentence_num_'])], axis=1)
    aggregate_per_sentence['log_prob_ldiff2'] = aggregate_per_sentence.apply(lambda x: stats['local_diff2'][x['text_id_']][int(x['sentence_num_'])], axis=1)
    aggregate_per_sentence['log_prob_std'] = np.sqrt(aggregate_per_sentence['log_prob_var'])
    return aggregate_per_sentence


# ## Datasets

# ### Natural Stories

# In[ ]:


gpt3_probs = pd.read_csv("https://raw.githubusercontent.com/languageMIT/naturalstories/master/probs/all_stories_gpt3.csv")
# To get same indexing as stories db
gpt3_probs["story"] = gpt3_probs["story"] + 1
gpt3_probs['len'] = gpt3_probs.groupby("story", sort=False)['offset'].shift(periods=-1, fill_value=0) - gpt3_probs['offset'] 
gpt3_probs['new_token'] = gpt3_probs.apply(lambda x: x['token'] if x['len'] == len(x['token']) else x['token'] + ' ', axis=1) 


# In[ ]:


stories_df = gpt3_probs.groupby(by=["story"], sort=False).agg({"new_token":[string_join]}).reset_index()
ns_stats = corpus_stats(zip(stories_df['story'], stories_df['new_token', 'string_join']), "gpt_probs.pkl")


# In[ ]:


import bisect
reading_times_df = pd.read_csv("https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS/processed_RTs.tsv", sep='\t').drop_duplicates()
reading_times_df.rename(columns = {'RT':'time', 
                                   'item': 'text_id'}, inplace = True)
reading_times_df['centered_time'] = reading_times_df['time'] - reading_times_df.groupby(by=["WorkerId"]).transform('mean')["time"]
reading_times_df['ref_token'] = reading_times_df.apply(lambda x: ns_stats['split_string'][x['text_id']][x['zone']-1], axis=1)
reading_times_df['prev_word'] = reading_times_df.apply(lambda x: ns_stats['split_string'][x['text_id']][x['zone']-2] if x['zone']-2 >= 0 else '', axis=1)
reading_times_df['log_prob'] = reading_times_df.apply(lambda x: ns_stats['agg_log_probs'][x['text_id']][x['zone']-1], axis=1)
reading_times_df['prev_log_prob'] = reading_times_df.apply(lambda x: ns_stats['agg_log_probs'][x['text_id']][x['zone']-2] if x['zone']-2 >= 0 else 0, axis=1)
reading_times_df['prev2_log_prob'] = reading_times_df.apply(lambda x: ns_stats['agg_log_probs'][x['text_id']][x['zone']-3] if x['zone']-3 >= 0 else 0, axis=1)
reading_times_df['prev3_log_prob'] = reading_times_df.apply(lambda x: ns_stats['agg_log_probs'][x['text_id']][x['zone']-4] if x['zone']-4 >= 0 else 0, axis=1)
reading_times_df['sentence_num'] = reading_times_df.apply(lambda x: bisect.bisect(ns_stats['sent_markers'][x['text_id']], x['zone']-1), axis=1)
reading_times_df['word_len'] = reading_times_df.apply(lambda x: len(x['word']), axis=1)
reading_times_df['prev_word_len'] = reading_times_df.apply(lambda x: len(x['prev_word']), axis=1)


# In[ ]:


# sanity check: looks like there's a small mispelling somewhere ;)
reading_times_df[reading_times_df['word'] != reading_times_df['ref_token']]


# In[ ]:


reading_times_df['freq'] = reading_times_df.apply(lambda x: root[()].get(x['word'],0), axis=1)
reading_times_df['prev_freq'] = reading_times_df.apply(lambda x: root[()].get(x['prev_word'],0), axis=1)
reading_times_df['diff'] = reading_times_df.apply(lambda x: x['log_prob']-ns_stats['means'][x['text_id']][x['sentence_num']], axis=1)
reading_times_df['var'] = reading_times_df.apply(lambda x: (x['log_prob']-ns_stats['means'][x['text_id']][x['sentence_num']])**2, axis=1)
reading_times_df['rolling_average'] = reading_times_df.sort_values(by='zone').groupby(by=["WorkerId","text_id", "sentence_num"])["log_prob"].transform(lambda x: x.expanding().mean())
reading_times_df['rolling_var'] = reading_times_df.apply(lambda x: (x['log_prob']-x['rolling_average'])**2, axis=1)


# In[ ]:


ns_agg_per_sentence = produce_aggregate_per_sentence(reading_times_df, ns_stats)


# ### Provo

# In[ ]:


provo = pd.read_csv('provo.csv')
provo.rename(columns = {'IA_DWELL_TIME':'time', 'Participant_ID': 'WorkerId', 
                        "Text_ID":"text_id", "Sentence_Number":"sentence_num"}, inplace = True)
provo = provo.dropna(subset=["Word_Number"])
provo = provo.astype({"Word_Number": 'Int64', "sentence_num": 'Int64'})
# First word isn't in dataset...
provo.loc[provo.Word_Number== 1]


# In[ ]:


def ordered_string_join(x, j=''):
    s = sorted(x, key=lambda x: x[0])
    a,b = list(zip(*s))
    return a, j.join(b)
inds, paragraphs = zip(*provo[['text_id','Word_Number','Word']].drop_duplicates().dropna().groupby(by = ['text_id']).apply(lambda x: ordered_string_join(zip(x['Word_Number'], x['Word']), ' ')))


# In[ ]:


provo_stats = corpus_stats(enumerate(paragraphs,1), "provo_gpt_probs.pkl")


# In[ ]:


provo['new_ind'] = provo.apply(lambda x: inds[x['text_id']-1].index(x["Word_Number"]), axis=1)
provo['centered_time'] = provo["time"] - provo.groupby(by=["WorkerId"]).transform('mean')["time"]
provo['sentence_num'] = provo['sentence_num'] - 1
provo['ref_token'] = provo.apply(lambda x: provo_stats['split_string'][x['text_id']][x['new_ind']], axis=1) 
provo['prev_word'] = provo.apply(lambda x: provo_stats['split_string'][x['text_id']][x['new_ind']-1] if x['new_ind'] > 0 else '', axis=1)
provo['log_prob'] = provo.apply(lambda x: provo_stats['agg_log_probs'][x['text_id']][x['new_ind']], axis=1)
provo['prev_log_prob'] = provo.apply(lambda x: provo_stats['agg_log_probs'][x['text_id']][x['new_ind']-1] if x['new_ind'] > 0 else 0, axis=1)
provo['prev2_log_prob'] = provo.apply(lambda x: provo_stats['agg_log_probs'][x['text_id']][x['new_ind']-2] if x['new_ind'] > 1 else 0, axis=1)
provo['prev3_log_prob'] = provo.apply(lambda x: provo_stats['agg_log_probs'][x['text_id']][x['new_ind']-3] if x['new_ind'] > 2 else 0, axis=1)
provo['word_len'] = provo.apply(lambda x: len(x['Word']), axis=1)
provo['prev_word_len'] = provo.apply(lambda x: len(x['prev_word']), axis=1)


# In[ ]:


provo['freq'] = provo.apply(lambda x: root[()].get(x['Word'],0), axis=1)
provo['prev_freq'] = provo.apply(lambda x: root[()].get(x['prev_word'],0), axis=1)


# In[ ]:


provo_agg_per_sentence = produce_aggregate_per_sentence(provo, provo_stats)


# ### UCL Reading

# In[ ]:


ucl = pd.read_csv('ucl/selfpacedreading.RT.txt','\t')
ucl.rename(columns = {'RT':'time', 'subj_nr': 'WorkerId', 
                        "sent_nr":"text_id"}, inplace = True)
ucl['sentence_num'] = 0


# In[ ]:


inds, paragraphs = zip(*ucl[['text_id','word_pos','word']].drop_duplicates().dropna().groupby(by = ['text_id']).apply(lambda x: ordered_string_join(zip(x['word_pos'], x['word']), ' ')))
ucl_stats = corpus_stats(enumerate(paragraphs,1), "ucl_gpt_probs.pkl")


# In[ ]:


ucl['new_ind'] = ucl.apply(lambda x: inds[x['text_id']-1].index(x["word_pos"]), axis=1)
# ref token is sanity check. should be same as word
ucl['ref_token'] = ucl.apply(lambda x: ucl_stats['split_string'][x['text_id']][x['new_ind']], axis=1) 
ucl['prev_word'] = ucl.apply(lambda x: ucl_stats['split_string'][x['text_id']][x['new_ind']-1] if x['new_ind'] > 0 else '', axis=1)
ucl['log_prob'] = ucl.apply(lambda x: ucl_stats['agg_log_probs'][x['text_id']][x['new_ind']], axis=1)
ucl['prev_log_prob'] = ucl.apply(lambda x: ucl_stats['agg_log_probs'][x['text_id']][x['new_ind']-1] if x['new_ind'] > 0 else 0, axis=1)
ucl['word_len'] = ucl.apply(lambda x: len(x['word']), axis=1)
ucl['prev_word_len'] = ucl.apply(lambda x: len(x['prev_word']), axis=1)
ucl['freq'] = ucl.apply(lambda x: root[()].get(x['word'],0), axis=1)
ucl['prev_freq'] = ucl.apply(lambda x: root[()].get(x['prev_word'],0), axis=1)


# In[ ]:


ucl_agg_per_sentence = produce_aggregate_per_sentence(ucl, ucl_stats)


# ### UCL Eye

# In[ ]:


ucl_eye = pd.read_csv('ucl/eyetracking.RT.txt','\t')
ucl_eye.rename(columns = {'RTfirstpass':'time', 'subj_nr': 'WorkerId', 
                        "sent_nr":"text_id"}, inplace = True)
ucl_eye['sentence_num'] = 0


# In[ ]:


joined = ucl_eye[['text_id','word_pos','word']].drop_duplicates().dropna().groupby(by = ['text_id']).apply(lambda x: ordered_string_join(zip(x['word_pos'], x['word']), ' '))
inds, paragraphs = zip(*joined)
ucl_eye_stats = corpus_stats(zip(joined.index, paragraphs), "ucl_eye_gpt_probs.pkl")


# In[ ]:


inds_dict = {i: ind_set for i, ind_set in zip(joined.index, inds)}
ucl_eye['new_ind'] = ucl_eye.apply(lambda x: inds_dict[x['text_id']].index(x["word_pos"]), axis=1)
# ref token is sanity check. should be same as word
ucl_eye['ref_token'] = ucl_eye.apply(lambda x: ucl_eye_stats['split_string'][x['text_id']][x['new_ind']], axis=1) 
ucl_eye['prev_word'] = ucl_eye.apply(lambda x: ucl_eye_stats['split_string'][x['text_id']][x['new_ind']-1] if x['new_ind'] > 0 else '', axis=1)
ucl_eye['log_prob'] = ucl_eye.apply(lambda x: ucl_eye_stats['agg_log_probs'][x['text_id']][x['new_ind']], axis=1)
ucl_eye['prev_log_prob'] = ucl_eye.apply(lambda x: ucl_eye_stats['agg_log_probs'][x['text_id']][x['new_ind']-1] if x['new_ind'] > 0 else 0, axis=1)
ucl_eye['word_len'] = ucl_eye.apply(lambda x: len(x['word']), axis=1)
ucl_eye['prev_word_len'] = ucl_eye.apply(lambda x: len(x['prev_word']), axis=1)
ucl_eye['freq'] = ucl_eye.apply(lambda x: root[()].get(x['word'],0), axis=1)
ucl_eye['prev_freq'] = ucl_eye.apply(lambda x: root[()].get(x['prev_word'],0), axis=1)


# In[ ]:


ucl_eye_agg_per_sentence = produce_aggregate_per_sentence(ucl_eye, ucl_eye_stats)


# ### CoLA

# In[ ]:


cola = pd.read_csv('cola_public/raw/in_domain_train.tsv','\t', header=None, names=['ID','accept','NA','sentence'])
cola = cola.drop(columns='NA')
cola_stats = corpus_stats(enumerate(cola['sentence']), "cola_gpt_probs.pkl")


# In[ ]:


cola['log_prob_sum'] = cola.apply(lambda x: np.sum(cola_stats['log_probs'][x.name][0]), axis=1)
cola['len'] = cola.apply(lambda x: len(cola_stats['split_string'][x.name]), axis=1)


# In[ ]:


cola['log_prob_mean'] = cola.apply(lambda x: cola_stats['means'][x.name][0], axis=1)
cola['log_prob_max'] = cola.apply(lambda x: cola_stats['maxes'][x.name][0], axis=1)
cola['log_prob_ent'] = cola.apply(lambda x: cola_stats['entropies'][x.name][0], axis=1)
cola['log_prob_var'] = cola.apply(lambda x: cola_stats['variances'][x.name][0], axis=1)
cola['log_prob_std'] = np.sqrt(cola['log_prob_var'])
names = []
power_range = np.arange(1, 4, 0.25)
for i in power_range:
    name = 'log_prob_power' + str(i)
    cola[name] = cola.apply(lambda x: np.sum(abs(cola_stats['log_probs'][x.name][0])**i), axis=1)
    names.append(name)


# In[ ]:


from sklearn.feature_selection import mutual_info_classif
for name in names:
    print(mutual_info_classif(np.array(cola[name]).reshape(-1, 1), cola['accept']))


# ## Psychometric Predictions

# In[ ]:


data = reading_times_df
aggregate_per_sentence = ns_agg_per_sentence


# In[ ]:


data = provo
aggregate_per_sentence = provo_agg_per_sentence


# In[ ]:


data = ucl
aggregate_per_sentence = ucl_agg_per_sentence


# ### Per Word

# In[ ]:


get_ipython().run_line_magic('R', '-i data')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'library(lme4)\nlibrary(MuMIn)')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'lme_cross_val <- function(formula, df, d_var, num_folds=10, shuffle=FALSE){\n    if(shuffle){\n        df <- df[sample(nrow(df)),]\n    }\n    folds <- cut(seq(1,nrow(df)),breaks=num_folds,labels=FALSE)\n    estimates <- c()\n    for(i in 1:num_folds){\n        testIndexes <- which(folds==i,arr.ind=TRUE)\n        testData <- df[testIndexes,]\n        trainData <- df[-testIndexes,]\n        model <- lmer(formula, REML=FALSE, data=trainData)\n        sigma <- mean(residuals(model)^2)\n        estimate <- log(dnorm(testData[[d_var]], \n                              mean=predict(model, newdata=testData, allow.new.levels=TRUE), \n                              sd=sqrt(sigma)))\n        estimates <- c(estimates, estimate)\n    }\n    estimates\n}')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'set.seed(42)\nshuffled_order <- sample(nrow(data))\npowers <- seq(1, 3.75, by=0.25)\n#baseline <- lme_cross_val("time ~ freq + word_len + (1 | WorkerId)", data[shuffled_order,], \'time\')\nout <- list()\nnames <- c(\'log_prob\', \'prev_log_prob\', \'prev2_log_prob\',\'prev3_log_prob\')\nfor(var in names){\n    other_vars <- paste(setdiff(names, var), collapse=" + ")\n    baseline <- lme_cross_val(paste0("time ~ ", other_vars, " + freq*word_len + (1 | WorkerId)"), data[shuffled_order,], \'time\')\n    power_func <- function(x){\n        data$log_prob_pow <- data[[var]]**x\n        formula <- paste0("time ~ log_prob_pow +", other_vars, " + freq*word_len  + (1 | WorkerId)")\n        cv <- lme_cross_val(formula, data[shuffled_order,], \'time\')\n        c(mean(cv-baseline, na.rm=TRUE), var(cv-baseline, na.rm=TRUE)/length(cv))\n    }\n    out[[var]] <- cbind(labels, as.data.frame(do.call(rbind,lapply(powers, power_func))))\n}')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'ns_baseline_out\nns_other_preds <- out')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'library(ggplot2)\nggplot(aes(x = labels, y = V1 ), data=ns_other_preds[[\'log_prob\']]) + \n    geom_line() +\n    geom_point(size=2) +\n    geom_ribbon(aes(ymin=V1-sqrt(V2), ymax=V1+sqrt(V2)), alpha = 0.2, fill=\'red\') +\n    ylab("Per Token Increase in LogLik") +\n    xlab("k") +\n    ggtitle("NS Corpus (with linear log-p predictors)") +\n    theme_minimal()')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'library(tidyr)\ncentered_df <- pivot_longer(data, c(\'log_prob\',\'prev_log_prob\', \'prev2_log_prob\',\'prev3_log_prob\'), names_to="pos", values_to="prob")\nggplot(aes(x=prob, y=centered_time, color=pos), data=centered_df) + \n    geom_smooth() ')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'formulas <- c("time~ log_prob + word_len + freq + (1 | WorkerId)",\n              "time~ log_prob  + prev_log_prob + freq + word_len+(1 | WorkerId)")\nbaseline <- lme_cross_val("time~ word_len + freq  +  (1 | WorkerId)", data[shuffled_order,], \'time\')\nfor(formula in formulas){\n    model <- lmer(formula, REML=FALSE, data=data)\n    print(logLik(model))\n    print(mean(lme_cross_val(formula, data[shuffled_order,], \'time\') - baseline, na.rm=TRUE))\n}')


# ### Per Sentence

# In[ ]:


get_ipython().run_line_magic('R', '-i aggregate_per_sentence')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'set.seed(42)\nshuffled_order <- aggregate_per_sentence[sample(nrow(aggregate_per_sentence)),]\nbaseline <- lme_cross_val("time_sum ~  time_count_nonzero  + (1 | WorkerId_)", \n                          aggregate_per_sentence[shuffled_order,],\n                         \'time_sum\')\nformulas <- c("time_sum ~  log_prob_mean + time_count_nonzero  + (1 | WorkerId_)",\n              "time_sum ~  log_prob_ldiff + time_count_nonzero  + (1 | WorkerId_)",\n              "time_sum ~  log_prob_max + time_count_nonzero  + (1 | WorkerId_)",\n              "time_sum ~  log_prob_var + time_count_nonzero  + (1 | WorkerId_)",\n             "time_sum ~  log_prob_ent + time_count_nonzero  + (1 | WorkerId_)")\nfor(formula in formulas){\n    model <- lmer(formula, REML=FALSE, data=aggregate_per_sentence)\n    print(logLik(model))\n    print(mean(lme_cross_val(formula, aggregate_per_sentence[shuffled_order,], \'time_sum\') - baseline, na.rm=TRUE))\n}\n\n#print(r.squaredGLMM(model)[1])')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'powers <- c("sum", "power1_25","power1_5","power1_75","power2","power2_25","power2_5","power2_75","power3","power3_25","power3_5","power3_75")\nlabels <- seq(1, 3.75, by=0.25)\nbaseline <- lme_cross_val_agg("time_sum ~  time_count_nonzero + (1 | WorkerId_)")\npower_func <- function(x){\n    formula <- paste0("time_sum ~ log_prob_", x," + time_count_nonzero  + (1 | WorkerId_)")\n    cv <- lme_cross_val_agg(formula)\n    c(mean(cv-baseline, na.rm=TRUE), var(cv-baseline, na.rm=TRUE)/length(cv))\n}\nout <- cbind(labels, as.data.frame(do.call(rbind,lapply(powers, power_func))))')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'library(ggplot2)\nggplot(aes(x = labels, y = V1 ), data=out) + \n    geom_line() +\n    geom_point(size=2) +\n    geom_ribbon(aes(ymin=V1-sqrt(V2), ymax=V1+sqrt(V2)), alpha = 0.2, fill=\'red\') +\n    ylab("Per Sentence Increase in LogLik") +\n    xlab("k") +\n    ggtitle("Provo Corpus") +\n    theme_minimal()')


# ## Corpus Analysis

# In[ ]:


import plotnine
from itertools import cycle


# In[ ]:


cola['accept'] = cola['accept'].astype('category')plotnine.ggplot() +     plotnine.aes(x='log_prob_mean', y="..density..", fill='accept') +     plotnine.geom_histogram(data=cola[cola['accept']=='1'], fill='blue', alpha = 0.5) +    plotnine.geom_histogram(data=cola[cola['accept']=='0'], fill='red', alpha = 0.5) +    plotnine.ylab("Density") +     plotnine.xlab("Mean Sentence Surprisal") +     plotnine.labels.ggtitle("CoLA corpus: blue = accept; red = reject") +     plotnine.scale_fill_manual(values = {'acceptable':'blue', 'not acceptable': 'red'})


# In[ ]:


cola.corr().accept


# In[ ]:


r = cola.corr().accept[names]
r_se = np.sqrt((1-r**2)/(len(cola) - 2))
get_ipython().run_line_magic('R', '-i r')
get_ipython().run_line_magic('R', '-i r_se')
get_ipython().run_line_magic('R', '-i power_range')


# In[ ]:


get_ipython().run_cell_magic('R', '', 'corrs <- as.data.frame(cbind(r,r_se,power_range))\nlibrary(ggplot2)\nggplot(aes(x = power_range, y = r ), data=corrs) + \n    geom_line() +\n    geom_point(size=2) +\n    geom_ribbon(aes(ymin=r-r_se, ymax=r+r_se), alpha = 0.2, fill=\'red\') +\n    theme_minimal() +\n    labs(x = "k", y="Pearson\'s correlation coef", title="Correlation between sentence acceptability judgement and sum surprisal^k")')

