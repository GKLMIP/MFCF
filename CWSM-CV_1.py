import numpy as np
from gensim.models import word2vec
import pickle
import gensim
import csv

model_loc = 'model/charmodel_LD_1'
with open('data/全部距离1候选词列表','rb') as f:
    candidates = pickle.load(f)
model = gensim.models.Word2Vec.load(model_loc)

whole, right= 0, 0
with open('data/Ind_1_70%_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:

        wrong_wo = row[1]
        if wrong_wo not in candidates:
            continue
        if len(candidates[wrong_wo]) != 0:
            wor_list = candidates[wrong_wo]
        else:
            continue

        word_vector = []
        vec = []
        for wor in wor_list:
            final_vec = model[wor[0]]
            i = 0
            for cha in wor:
                if i == 0:
                    i += 1
                    continue
                final_vec = final_vec + model[cha]
            final_vec = final_vec / len(wor)
            vec.append(final_vec)
            word_vector.append(wor)

        vec = np.array(vec)
        wrong_word_vec = model[wrong_wo[0]]
        i = 0
        for cha in wrong_wo:
            if i == 0:
                i += 1
                continue
            if cha not in model:
                continue
            wrong_word_vec = wrong_word_vec + model[cha]
        wrong_word_vec = wrong_word_vec / len(wrong_wo)

        rank = {}
        rank_cos = []
        for i in range(len(word_vector)):
            dot = np.dot(vec[i], wrong_word_vec)
            norma = np.linalg.norm(vec[i])
            normb = np.linalg.norm(wrong_word_vec)
            cos = dot / (norma * normb)
            rank[cos] = word_vector[i]
            rank_cos.append(cos)
        rank_cos = sorted(rank_cos, reverse=True)

        whole += 1
        result = []
        for nu in rank_cos:
            result.append(rank[nu])
        if row[2] in result[:1]:
            right += 1

print(right)
print(whole)
num_rate = right/whole
print('正确率：'+str(num_rate))



