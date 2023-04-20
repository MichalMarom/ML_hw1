# ----------- File For Reading Data-----------
import readTrainData as Rdata
# ----------- Python Package -----------
from collections import Counter
import numpy as np


def learn_NB_text():
    texAll, lbAll, voc, cat = Rdata.readTrainData("r8-train-stemmed.txt")
    cat = list(cat)
    cat.sort()
    voc = list(voc)
    voc.sort()

    pr_w = posterior(texAll, lbAll, voc, cat, alpha=1)  # p(Xj|Wi)
    p = calc_p(cat, lbAll)

    # voc_cat_prob = {a: 0 for a in cat}
    # print(voc_cat_prob)
    # for i in lbAll:
    #     voc_cat_prob[i] += 1
    #
    # print(voc_cat_prob)
    #
    # for i in voc_cat_prob:
    #     voc_cat_prob[i] = voc_cat_prob[i] / len(lbAll)
    #
    # print(voc_cat_prob)
    return pr_w, p


def posterior(data: list, labeled_data: list, words: list, categories: list, alpha):

    pr_w = {}
    words_in_categories = [list()] * len(categories)  # List of all words in each category

    for i in range(len(categories)):  # For every category
        for j in range(len(labeled_data)):
            if categories[i] == labeled_data[j]:  # If the label of the sentence equal to category
                words_in_categories[i] = words_in_categories[i] + data[j]  # Add the words to the category list

    for i, cat in enumerate(categories):
        count_words_in_cat = Counter(words_in_categories[i])
        pr_w[cat] = {word: (count_words_in_cat[words[j]] + alpha) / (len(words_in_categories[i]) + (alpha * len(words))) for j, word in enumerate(words)}
        pr_w[cat]["unknown_word"] = alpha / (len(words_in_categories[i]) + alpha * len(cat))
        
    # pr_w = np.zeros((len(categories), len(words)+1)) # +1 for unknow words
    # cat_counts = Counter(labeled_data)
    # for i in range(len(categories)):
    #     words_count_for_cat = Counter(words_in_cat[i])
    #     for j in range(len(words)):
    #         pr_w[i][j] = (words_count_for_cat[words[i]] + alpha) / (cat_counts[categories[i]] + (alpha * len(words)))
    #         # if words_count_for_cat[words[i]] == 0:
    #         #     pr_w[i][j] = (words_count_for_cat[words[i]] + alpha) / (cat_counts[categories[i]] + (alpha * len(words)))
    #         # else:
    #         #     pr_w[i][j] = words_count_for_cat[words[i]] / cat_counts[categories[i]]
    #     pr_w[i][len(words)] = alpha / (cat_counts[categories[i]] + (alpha * len(words)))

        # for i in range(len(categories)):
        #     words_count_for_cat = Counter(words_in_class[i])
        #     for j in range(len(words)):
        #         counter = 0
        #         for index in range(len(data)):
        #             if labeled_data[index] == categories[i] and words[j] in data[index]:
        #                 counter += 1
        #         if counter == 0:
        #             pr_w[i][j] = (counter + alpha) / (cat_counts[categories[i]] + (alpha * len(words)))
        #         else:
        #             pr_w[i][j] = counter / cat_counts[categories[i]]
        #
        #     pr_w[i][len(words)] = alpha / (cat_counts[categories[i]] + (alpha * len(words)))

    return pr_w


def calc_p(categories: list, data: list):
    appearance = []
    word_counts = Counter(data)

    for word in categories:
        appearance.append(word_counts[word] / len(data))

    return appearance


def ClassifyNB_text(Pw, P):
    texAll, lbAll, _, _ = Rdata.readTrainData("r8-test-stemmed.txt")
    _, _, voc, cat = Rdata.readTrainData("r8-train-stemmed.txt")
    cat = list(cat)
    cat.sort()
    voc = list(voc)
    voc.sort()
    suc = 0

    for index, sentence in enumerate(texAll):
        label_sentence = lbAll[index]
        categories_mul = [np.log(P[i]) + sum(np.log(Pw[category][word]) for word in sentence if word in Pw[category]) for i, category in enumerate(cat)]
        max_labels_index = categories_mul.index(max(categories_mul))
        if cat[max_labels_index] == label_sentence:
            suc += 1

        # for i, category in enumerate(cat):
        #     cat_mul = 0
        #     for word in Pw[category]:
        #         if word in voc:
        #             cat_mul += np.log(Pw[category][word])
        #         else:
        #             cat_mul += np.log(Pw[category]["unknown_word"])
        #     cat_mul += np.log(P[i])


            # cat_mul = word_in_sentence * Pw[i, 0:len(voc)] + (1 - word_in_sentence) * (1 - Pw[i, 0:len(voc)])
            # cat_mul = np.sum(np.log(cat_mul)) + np.log(P[i])
            # cat_mul += count_words_not_in_voc * np.log(Pw[i][len(voc)])
            # word_in_sentence = np.array([1 * (word in sentence) for word in Pw[category]])
            # cat_mul = word_in_sentence * Pw[i].values() + (1 - word_in_sentence) * (1 - Pw[i].values())
            # cat_mul = np.sum(np.log(cat_mul)) + np.log(P[i])
            # categories_mul.append(cat_mul)
        # max_labels_index = categories_mul.index(max(categories_mul))
        # if cat[max_labels_index] == label_sentence:
        #      suc += 1

    return suc / len(texAll)


if __name__ == '__main__':
    Pw, P = learn_NB_text()
    np.save(file='Pw.npy', arr=Pw)
    np.save(file='P.npy', arr=np.asarray(P))
    x = 2
    print("end Pw, P ")
    # Pw = np.load('Pw.npy')
    # P = np.load('P.npy')
    suc = ClassifyNB_text(Pw, P)
    print(suc)

    # np.save(file='Pw.npy', arr=Pw)
    # np.save(file='P.npy', arr=np.asarray(P))
