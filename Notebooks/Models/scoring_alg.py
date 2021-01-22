import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt

def compute_challenge_metric_for_opt(labels, outputs):
    classes=['10370003','111975006','164889003','164890007','164909002','164917005','164934002','164947007','17338001',
 '251146004','270492004','284470004','39732003','426177001','426627000','426783006','427084000','427172004','427393009','445118002','47665007','59118001',
 '59931005','63593006','698252002','713426002','713427006']


    normal_class = '426783006'
    weights = np.array([[1.    , 0.425 , 0.375 , 0.375 , 0.4   , 0.275 , 0.375 , 0.425 ,
        0.5   , 0.45  , 0.425 , 0.4625, 0.475 , 0.425 , 0.425 , 0.375 ,
        0.5   , 0.5   , 0.425 , 0.475 , 0.475 , 0.475 , 0.375 , 0.4625,
        0.475 , 0.425 , 0.475 ],
       [0.425 , 1.    , 0.45  , 0.45  , 0.475 , 0.35  , 0.45  , 0.35  ,
        0.425 , 0.475 , 0.35  , 0.3875, 0.4   , 0.35  , 0.35  , 0.3   ,
        0.425 , 0.425 , 0.35  , 0.4   , 0.4   , 0.45  , 0.45  , 0.3875,
        0.4   , 0.35  , 0.45  ],
       [0.375 , 0.45  , 1.    , 0.5   , 0.475 , 0.4   , 0.5   , 0.3   ,
        0.375 , 0.425 , 0.3   , 0.3375, 0.35  , 0.3   , 0.3   , 0.25  ,
        0.375 , 0.375 , 0.3   , 0.35  , 0.35  , 0.4   , 0.5   , 0.3375,
        0.35  , 0.3   , 0.4   ],
       [0.375 , 0.45  , 0.5   , 1.    , 0.475 , 0.4   , 0.5   , 0.3   ,
        0.375 , 0.425 , 0.3   , 0.3375, 0.35  , 0.3   , 0.3   , 0.25  ,
        0.375 , 0.375 , 0.3   , 0.35  , 0.35  , 0.4   , 0.5   , 0.3375,
        0.35  , 0.3   , 0.4   ],
       [0.4   , 0.475 , 0.475 , 0.475 , 1.    , 0.375 , 0.475 , 0.325 ,
        0.4   , 0.45  , 0.325 , 0.3625, 0.375 , 0.325 , 0.325 , 0.275 ,
        0.4   , 0.4   , 0.325 , 0.375 , 0.375 , 0.425 , 0.475 , 0.3625,
        0.375 , 0.325 , 0.425 ],
       [0.275 , 0.35  , 0.4   , 0.4   , 0.375 , 1.    , 0.4   , 0.2   ,
        0.275 , 0.325 , 0.2   , 0.2375, 0.25  , 0.2   , 0.2   , 0.15  ,
        0.275 , 0.275 , 0.2   , 0.25  , 0.25  , 0.3   , 0.4   , 0.2375,
        0.25  , 0.2   , 0.3   ],
       [0.375 , 0.45  , 0.5   , 0.5   , 0.475 , 0.4   , 1.    , 0.3   ,
        0.375 , 0.425 , 0.3   , 0.3375, 0.35  , 0.3   , 0.3   , 0.25  ,
        0.375 , 0.375 , 0.3   , 0.35  , 0.35  , 0.4   , 0.5   , 0.3375,
        0.35  , 0.3   , 0.4   ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 1.    ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 0.5   , 0.5   , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.5   , 0.425 , 0.375 , 0.375 , 0.4   , 0.275 , 0.375 , 0.425 ,
        1.    , 0.45  , 0.425 , 0.4625, 0.475 , 0.425 , 0.425 , 0.375 ,
        0.5   , 1.    , 0.425 , 0.475 , 0.475 , 0.475 , 0.375 , 0.4625,
        0.475 , 0.425 , 0.475 ],
       [0.45  , 0.475 , 0.425 , 0.425 , 0.45  , 0.325 , 0.425 , 0.375 ,
        0.45  , 1.    , 0.375 , 0.4125, 0.425 , 0.375 , 0.375 , 0.325 ,
        0.45  , 0.45  , 0.375 , 0.425 , 0.425 , 0.475 , 0.425 , 0.4125,
        0.425 , 0.375 , 0.475 ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 1.    , 0.4625, 0.45  , 0.5   , 0.5   , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.4625, 0.3875, 0.3375, 0.3375, 0.3625, 0.2375, 0.3375, 0.4625,
        0.4625, 0.4125, 0.4625, 1.    , 0.4875, 0.4625, 0.4625, 0.4125,
        0.4625, 0.4625, 0.4625, 0.4875, 0.4875, 0.4375, 0.3375, 1.    ,
        0.4875, 0.4625, 0.4375],
       [0.475 , 0.4   , 0.35  , 0.35  , 0.375 , 0.25  , 0.35  , 0.45  ,
        0.475 , 0.425 , 0.45  , 0.4875, 1.    , 0.45  , 0.45  , 0.4   ,
        0.475 , 0.475 , 0.45  , 0.5   , 0.5   , 0.45  , 0.35  , 0.4875,
        0.5   , 0.45  , 0.45  ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 1.    , 0.5   , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 0.5   , 1.    , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.375 , 0.3   , 0.25  , 0.25  , 0.275 , 0.15  , 0.25  , 0.45  ,
        0.375 , 0.325 , 0.45  , 0.4125, 0.4   , 0.45  , 0.45  , 1.    ,
        0.375 , 0.375 , 0.45  , 0.4   , 0.4   , 0.35  , 0.25  , 0.4125,
        0.4   , 0.45  , 0.35  ],
       [0.5   , 0.425 , 0.375 , 0.375 , 0.4   , 0.275 , 0.375 , 0.425 ,
        0.5   , 0.45  , 0.425 , 0.4625, 0.475 , 0.425 , 0.425 , 0.375 ,
        1.    , 0.5   , 0.425 , 0.475 , 0.475 , 0.475 , 0.375 , 0.4625,
        0.475 , 0.425 , 0.475 ],
       [0.5   , 0.425 , 0.375 , 0.375 , 0.4   , 0.275 , 0.375 , 0.425 ,
        1.    , 0.45  , 0.425 , 0.4625, 0.475 , 0.425 , 0.425 , 0.375 ,
        0.5   , 1.    , 0.425 , 0.475 , 0.475 , 0.475 , 0.375 , 0.4625,
        0.475 , 0.425 , 0.475 ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 0.5   , 0.5   , 0.45  ,
        0.425 , 0.425 , 1.    , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 0.5   , 0.4   ],
       [0.475 , 0.4   , 0.35  , 0.35  , 0.375 , 0.25  , 0.35  , 0.45  ,
        0.475 , 0.425 , 0.45  , 0.4875, 0.5   , 0.45  , 0.45  , 0.4   ,
        0.475 , 0.475 , 0.45  , 1.    , 0.5   , 0.45  , 0.35  , 0.4875,
        0.5   , 0.45  , 0.45  ],
       [0.475 , 0.4   , 0.35  , 0.35  , 0.375 , 0.25  , 0.35  , 0.45  ,
        0.475 , 0.425 , 0.45  , 0.4875, 0.5   , 0.45  , 0.45  , 0.4   ,
        0.475 , 0.475 , 0.45  , 0.5   , 1.    , 0.45  , 0.35  , 0.4875,
        0.5   , 0.45  , 0.45  ],
       [0.475 , 0.45  , 0.4   , 0.4   , 0.425 , 0.3   , 0.4   , 0.4   ,
        0.475 , 0.475 , 0.4   , 0.4375, 0.45  , 0.4   , 0.4   , 0.35  ,
        0.475 , 0.475 , 0.4   , 0.45  , 0.45  , 1.    , 0.4   , 0.4375,
        0.45  , 0.4   , 1.    ],
       [0.375 , 0.45  , 0.5   , 0.5   , 0.475 , 0.4   , 0.5   , 0.3   ,
        0.375 , 0.425 , 0.3   , 0.3375, 0.35  , 0.3   , 0.3   , 0.25  ,
        0.375 , 0.375 , 0.3   , 0.35  , 0.35  , 0.4   , 1.    , 0.3375,
        0.35  , 0.3   , 0.4   ],
       [0.4625, 0.3875, 0.3375, 0.3375, 0.3625, 0.2375, 0.3375, 0.4625,
        0.4625, 0.4125, 0.4625, 1.    , 0.4875, 0.4625, 0.4625, 0.4125,
        0.4625, 0.4625, 0.4625, 0.4875, 0.4875, 0.4375, 0.3375, 1.    ,
        0.4875, 0.4625, 0.4375],
       [0.475 , 0.4   , 0.35  , 0.35  , 0.375 , 0.25  , 0.35  , 0.45  ,
        0.475 , 0.425 , 0.45  , 0.4875, 0.5   , 0.45  , 0.45  , 0.4   ,
        0.475 , 0.475 , 0.45  , 0.5   , 0.5   , 0.45  , 0.35  , 0.4875,
        1.    , 0.45  , 0.45  ],
       [0.425 , 0.35  , 0.3   , 0.3   , 0.325 , 0.2   , 0.3   , 0.5   ,
        0.425 , 0.375 , 0.5   , 0.4625, 0.45  , 0.5   , 0.5   , 0.45  ,
        0.425 , 0.425 , 0.5   , 0.45  , 0.45  , 0.4   , 0.3   , 0.4625,
        0.45  , 1.    , 0.4   ],
       [0.475 , 0.45  , 0.4   , 0.4   , 0.425 , 0.3   , 0.4   , 0.4   ,
        0.475 , 0.475 , 0.4   , 0.4375, 0.45  , 0.4   , 0.4   , 0.35  ,
        0.475 , 0.475 , 0.4   , 0.45  , 0.45  , 1.    , 0.4   , 0.4375,
        0.45  , 0.4   , 1.    ]])
    
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = float('nan')

    return normalized_score

def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A


def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    macro_f_measure = np.nanmean(f_measure)

    return macro_f_measure

# Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
def compute_beta_measures(labels, outputs, beta):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs, normalize=True)

    f_beta_measure = np.zeros(num_classes)
    g_beta_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if (1+beta**2)*tp + fp + beta**2*fn:
            f_beta_measure[k] = float((1+beta**2)*tp) / float((1+beta**2)*tp + fp + beta**2*fn)
        else:
            f_beta_measure[k] = float('nan')
        if tp + fp + beta*fn:
            g_beta_measure[k] = float(tp) / float(tp + fp + beta*fn)
        else:
            g_beta_measure[k] = float('nan')

    macro_f_beta_measure = np.nanmean(f_beta_measure)
    macro_g_beta_measure = np.nanmean(g_beta_measure)

    return macro_f_beta_measure, macro_g_beta_measure

# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :]==outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)

# Compute confusion matrices.
def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A


def compute_modified_confusion_matrix_nonorm(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        #####normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0#/normalization

    return A


def plot_classes(classes,snomedscored,y_label):
    for j in range(len(classes)):
        for i in range(len(snomedscored.iloc[:,1])):
            if (str(snomedscored.iloc[:,1][i]) == classes[j]):
                classes[j] = snomedscored.iloc[:,0][i]
    plt.style.use(['seaborn-paper'])      
    plt.figure(figsize=(30,20))
    plt.bar(x=classes,height=y_label.sum(axis=0),color="black")
    #plt.title("Distribution of Diagnosis", color = "black")
    plt.tick_params(axis="both", colors = "black")
    plt.xlabel("Diagnosis", color = "black",fontsize = 40)
    plt.ylabel("Count", color = "black", fontsize=40)
    plt.xticks(rotation=90, fontsize=40)
    plt.yticks(fontsize = 40)
    plt.savefig("diagnoses_distribution.png",dpi=200,bbox_inches = 'tight')
    plt.show()
    

def plot_classes_2(classes,y_label,snomedscored, norskliste):

    for j in range(len(classes)):
        for i in range(len(snomedscored.iloc[:,1])):
            if (str(snomedscored.iloc[:,1][i]) == classes[j]):
                classes[j] = snomedscored.iloc[:,0][i]
    plt.figure(figsize=(20,10))
    plt.gca()
    plt.bar(x=norskliste,height=y_label.sum(axis=0))
    #plt.title("Distribution of Diagnosis", color = "black")
    plt.tick_params(axis="both", colors = "black")
    plt.xlabel("Diagnoser", color = "black", fontsize=20)
    plt.ylabel("Antall", color = "black", fontsize=20)

    plt.xticks(rotation=90, fontsize=20)
    #plt.grid()
    plt.yticks(fontsize = 20)
    plt.savefig("fordeling.png",dpi=200,bbox_inches = 'tight')
    plt.show()
