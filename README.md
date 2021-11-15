# Multilevel Characterization of Protein Sequences


In this work Multilevel Protein Prediction Model is proposed for the category, characteristics, species, and function characterization of known and protein sequences. Four levels of prediction are performed in ML_PP model.  Protein sequences have been characterized as Gram-Positive Protein Sequences (G+_PS) and Non-Gram-Positive Protein Sequences (Non_G+_PS) at the first level of prediction, for the further levels of prediction G+_PS sequences are considered while Non_G+_PS sequences are discarded. In the next level G+_PS are characterized as Pathogenic (Path_G+) and Nonpathogenic (Non_Path_G+). Three different species such as Staphylococcus Aureus (Path_G+_SP1), Clostridia (Path_G+_SP2), and Streptococcus Pneumoniae (Path_G+_SP3) of the Path_G+ protein sequences are identified in the third level of prediction. In the fourth level functions of the three different Path_G+ species have been characterized with reference to the specified Protein Subcellular Localization (PSL) of the sequence. Performance of the ML_PP model is validated in each level of the prediction using known protein sequences with 5-fold and 10-fold cross validation as well as accuracy, precision, recall, and f1-score. Validated ML_PP model has been further utilized for the category, characteristics, species, and functions prediction of unknown protein sequences.

Folder “Model Training and Validation using Known Protein Data” contains the codes and datasets for all levels (Level 1, Level 2, Level 3, Level 4a, Level 4b, Level 4c) of the prediction for known protein sequences. Folder “Prediction of Unknown Protein Sequences” contains codes and dataset for all levels of the prediction for unknown protein sequence. 
