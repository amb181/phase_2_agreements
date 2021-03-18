# Machine Learning using scikit-learn library.

Selected model: Linear Support Vector Classification.

Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.


These models are used to extract different sections of a specific class, datasets were csv files with one column with the text to detect and other column with its class.
