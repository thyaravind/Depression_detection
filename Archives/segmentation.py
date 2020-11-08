from pyAudioAnalysis import audioSegmentation as aS
[flagsInd, classesAll, acc, CM] = aS.mid_term_file_classification("303.wav", "data/models/svm_rbf_sm", "svm", True, 'data/scottish.segments')