import readTrainData as Rdata
from collections import Counter
import numpy as np
import math


def learn_NB_text():
    texAll, lbAll, voc, cat = Rdata.readTrainData("r8-train-stemmed.txt")
    cat = list(cat)
    cat.sort()
    voc = list(voc)
    voc.sort()
    # pr_words = pr_of_appearance_in_matrix(voc, texAll)  # p(Xi)
    # pr_categories = pr_of_appearance_in_list(cat, lbAll)  # p(categories)
    # likelihood_matrix = likelihood(voc, cat, texAll, lbAll, pr_words)  # p(Wi|Xj)
    # np.save(file='likelihood_matrix.npy', arr=likelihood_matrix)
    # likelihood_matrix = np.load('likelihood_matrix.npy')
    # pr_categories = find_pr_categories(likelihood_matrix, pr_words)  # p(categories)
    pr_w = posterior(texAll, lbAll, voc, cat, alpha=1)  # p(Xj|Wi)
    p = pr_of_appearance_in_list(cat, lbAll)

    return pr_w, p


def posterior(data: list, labeled_data: list, words: list, categories: list, alpha):

    pr_w = np.zeros((len(categories), len(words)))
    cat_counts = Counter(labeled_data)

    for i in range(len(categories)):
        for j in range(len(words)):
            counter = 0
            for index in range(len(data)):
                if labeled_data[index] == categories[i] and words[j] in data[index]:
                    counter += 1
            pr_w[i][j] = (counter + alpha) / (cat_counts[categories[i]] + (alpha * len(words)))
            # if counter == 0:
            #
            # else:
            #     pr_w[i][j] = counter / cat_counts[categories[i]]
    return pr_w


def pr_of_appearance_in_list(categories: list, data: list):
    appearance = []
    word_counts = Counter(data)

    for word in categories:
        appearance.append(word_counts[word] / len(data))
        # appearance.append((word_counts[word] + 1) / (len(data) + len(categories)))
        # if word_counts[word] == 0:
        #     appearance.append((word_counts[word] + 1) / (len(data) + 2))
        # else:
        #     appearance.append(word_counts[word] / len(data))

    return appearance


# def pr_of_appearance_in_matrix(categories: list, data: list):
#     appearance = []
#     size_data = [len(data[index]) for index in range(len(data))]
#     size_data = sum(size_data)
#
#     for word in categories:
#         counter = 0
#         for index in range(len(data)):
#             counter += data[index].count(word)
#         appearance.append(counter/size_data)
#
#     # sum_appearance = sum(appearance)
#     # for i in range(len(appearance)):
#     #     appearance[i] = appearance[i] / sum_appearance
#
#     return appearance


# def likelihood(words: list, categories: list, data: list, labeled_data: list, pr_words: list):
#     likelihood_matrix = np.zeros([len(words), len(categories)])
#     size_data = [len(data[index]) for index in range(len(data))]
#     size_data = sum(size_data)
#
#     for i, word in enumerate(words):
#         counter = 0
#         for j, cats in enumerate(categories):
#             for index in range(len(data)):
#                 if labeled_data[index] == cats:
#                     counter += data[index].count(word)
#             # likelihood_matrix[i][j] = (counter/size_data) * pr_words[i] # p(category given word) * p(appearance of word)
#             likelihood_matrix[i][j] = (counter / size_data)
#     return likelihood_matrix
#
#
# def find_pr_categories(likelihood_matrix, pr_words: list):
#     pr_cat = []
#     for cat in range(likelihood_matrix.shape[1]):
#         complete_pr = 0
#         for word in range(likelihood_matrix.shape[0]):
#             complete_pr += likelihood_matrix[word][cat]*pr_words[word]
#         pr_cat.append(complete_pr)
#
#     return pr_cat


# def posterior(pr_words: list, pr_categories: list, likelihood_matrix: list):
#     pr_w = np.zeros([len(pr_words), len(pr_categories)])
#     for i in range(len(pr_words)):
#         for j in range(len(pr_categories)):
#             post = (likelihood_matrix[i][j] * pr_words[i]) / pr_categories[j]
#             pr_w[i][j] = post
#
#     return pr_w


def ClassifyNB_text(Pw, P):
    texAll, lbAll, _, cat = Rdata.readTrainData("r8-test-stemmed.txt")
    _, _, voc, _ = Rdata.readTrainData("r8-train-stemmed.txt")
    cat = list(cat)
    cat.sort()
    voc = list(voc)
    voc.sort()
    # arg_max = arg_max_word(Pw, P, voc, cat)
    suc = 0

    for index, sentence in enumerate(texAll):
        label_sentence = lbAll[index]
        categories_mul = []
        for i, category in enumerate(cat):
            word_in_sentence = np.array([1 * (word in sentence) for word in voc])
            cat_mul = word_in_sentence * Pw[i, :] + (1 - word_in_sentence) * (1 - Pw[i, :])
            cat_mul = np.sum(np.log(cat_mul)) + np.log(P[i])
            categories_mul.append(cat_mul)
        max_labels_index = categories_mul.index(max(categories_mul))
        if cat[max_labels_index] == label_sentence:
            suc += 1

    return suc/len(texAll)


def arg_max_word(Pw, P, words: list, categories: list):
    arg_max = {}

    for index, word in enumerate(words):
        arg_mul = [Pw[j][index] * P[j] for j in range(len(categories))]
        max_class_index = arg_mul.index(max(arg_mul))
        arg_max[word] = categories[max_class_index]

    return arg_max


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
