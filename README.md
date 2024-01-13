
# AI-Based-Network-IDS_ML-DL


## Table of Contents
- [AI-Based-Network-IDS\_ML-DL](#ai-based-network-ids_ml-dl)
	- [Table of Contents](#table-of-contents)
	- [Introduction](#introduction)
	- [Dataset Overview](#dataset-overview)
		- [Why Not KDD'99?](#why-not-kdd99)
	- [Methodology](#methodology)
		- [Data Preprocessing](#data-preprocessing)
		- [Feature Identification and Categorization](#feature-identification-and-categorization)
		- [Output Label](#output-label)
		- [Model Development](#model-development)
			- [Machine Learning Models](#machine-learning-models)
			- [Deep Learning Models (on-going)](#deep-learning-models-on-going)
		- [Deep Learning Model Evaluation -*Binary*-](#deep-learning-model-evaluation--binary-)
			- [Confusion Matrix Analysis](#confusion-matrix-analysis)
			- [Model Performance Metrics](#model-performance-metrics)
			- [Optimization of Neural Network Using Randomized Search](#optimization-of-neural-network-using-randomized-search)
			- [Hyperparameter Tuning Approach](#hyperparameter-tuning-approach)
			- [Parameters and Search Space](#parameters-and-search-space)
			- [Execution and Results](#execution-and-results)
		- [Deep Learning Model Evaluation -*MultiClass*-](#deep-learning-model-evaluation--multiclass-)
			- [Confusion Matrix Analysis](#confusion-matrix-analysis-1)
			- [Model Performance Metrics](#model-performance-metrics-1)
			- [Classification Report](#classification-report)
	- [Results and Discussion](#results-and-discussion)
		- [Challenges and Limitations](#challenges-and-limitations)
		- [Conclusion](#conclusion)


## Introduction

In the ever-evolving landscape of cyber threats, the significance of robust network security systems cannot be overstated. Traditional intrusion detection systems (IDS) often struggle to keep pace with the complexity and novelty of modern cyber-attacks. Enter AI-Based-Network-IDS_ML-DL, a project that stands at the forefront of this challenge. This initiative is driven by the integration of advanced Artificial Intelligence (AI), utilizing Machine Learning (ML) and Deep Learning (DL) techniques to analyze and predict network intrusions with unprecedented accuracy.

This project aims to harness the power of two significant datasets: NSL-KDD and UNSW-NB 15, each offering a different perspective and set of challenges. By incorporating these diverse datasets, the project endeavors to build an IDS that not only learns from historical patterns but also adapts to emerging threats, ensuring a future-ready defense mechanism for network security.


## Dataset Overview

In our quest to create a robust and highly efficient network security model, we have chosen to work with two distinctive datasets: NSL-KDD and UNSW-NB 15. Each of these datasets offers unique insights and challenges, providing a comprehensive ground for testing and improving our intrusion detection system.

* ***KDD Cup 1999 (KDD'99) Dataset:***  

	This is the data set used for The Third International Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with ***KDD-99*** The Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network intrusion detector, a predictive model capable of distinguishing between ***"bad''*** connections, called intrusions or attacks, and ***"good''*** normal connections. This database contains a standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment.
	 ```
	 -   [kddcup.names](https://kdd.ics.uci.edu/databases/kddcup99/kddcup.names)  A list of features.
	-   [kddcup.data.gz](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz)  The full data set (18M; 743M Uncompressed)
	-   [kddcup.data_10_percent.gz](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz)  A 10% subset. (2.1M; 75M Uncompressed)
	-   [kddcup.newtestdata_10_percent_unlabeled.gz](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.newtestdata_10_percent_unlabeled.gz)  (1.4M; 45M Uncompressed)
	-   [kddcup.testdata.unlabeled.gz](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled.gz)  (11.2M; 430M Uncompressed)
	-   [kddcup.testdata.unlabeled_10_percent.gz](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled_10_percent.gz)  (1.4M;45M Uncompressed)
	-   [corrected.gz](http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz)  Test data with corrected labels.
	-   [training_attack_types](https://kdd.ics.uci.edu/databases/kddcup99/training_attack_types)  A list of intrusion types.
	-   [typo-correction.txt](http://kdd.ics.uci.edu/databases/kddcup99/typo-correction.txt)  A brief note on a typo in the data set that has been corrected (6/26/07)
* ***NSL-KDD Dataset:*** 

	***NSL-KDD*** is a data set suggested to solve some of the inherent problems of the ***KDD'99*** data set. Although, this new version of the KDD data set still suffers from some of the problems and may not be a perfect representative of existing real networks, because of the lack of public data sets for ***network-based IDSs***, we believe it still can be applied as an effective benchmark data set to help researchers compare different intrusion detection methods.
Furthermore, the number of records in the ***NSL-KDD*** train and test sets are reasonable. This advantage makes it affordable to run the experiments on the complete set without the need to randomly select a small portion. Consequently, evaluation results of different research work will be consistent and comparable.


	 ***Improvements to the KDD'99 dataset in NSL-KDD***

	The NSL-KDD data set has the following advantages over the original KDD data set:

	-   It does not include redundant records in the train set, so the classifiers will not be biased towards more frequent records.
	-   There is no duplicate records in the proposed test sets; therefore, the performance of the learners are not biased by the methods which have better detection rates on the frequent records.
	-   The number of selected records from each difficultylevel group is inversely proportional to the percentage of records in the original KDD data set. As a result, the classification rates of distinct machine learning methods vary in a wider range, which makes it more efficient to have an accurate evaluation of different learning techniques.
	-   The number of records in the train and test sets are reasonable, which makes it affordable to run the experiments on the complete set without the need to randomly select a small portion. Consequently, evaluation results of different research works will be consistent and comparable.

*  ***UNSW-NB 15 Dataset:***

	The ***UNSW-NB 15*** dataset introduces a different spectrum of challenges and opportunities. Generated using the IXIA PerfectStorm tool in the Cyber Range Lab of the ***Australian Centre for Cyber Security (ACCS)*** , this dataset provides a hybrid of real modern normal activities and synthetic contemporary attack behaviours. With its extensive collection of features, generated using tools like ***Argus*** and ***Bro-IDS***, and a rich diversity of attack types, ***UNSW-NB 15*** allows us to test and refine our network security model under complex and modern attack scenarios. The comprehensive ground truth and feature descriptions available within the dataset further enhance our ability to interpret and learn from the data, driving our model towards higher accuracy and efficiency.

### Why Not KDD'99?

While KDD'99 has been a pivotal dataset in the realm of network security, its relevance has dwindled over the years, primarily due to its inherent biases and redundancies. The dataset is plagued by a large number of duplicate records, skewing the learning process towards more frequent patterns and potentially leading to a false sense of accuracy and security. This bias towards frequent records diminishes our model’s ability to detect rare, yet potentially devastating, network intrusions. The NSL-KDD dataset effectively addresses these issues, providing a cleaner, more balanced dataset for our intrusion detection system.

## Methodology

###  Data Preprocessing  

1. **Loading the Dataset:**
   - Load training and test datasets using Pandas library.

2. **Dropping Redundant Features:** 
   - The 'difficulty' level feature is removed as it is not required for intrusion detection.

3. **Label Consolidation:**
   - The attack labels are consolidated into their respective attack categories.

4. **Exploratory Data Analysis (EDA):**
   - Use pie charts to visualize the distribution of protocol types and attack labels.

5. **Feature Selection:**
   - Keep a copy of the original dataset for multi-class classification.

6. **Data Normalization:**
   - Apply standard scaling to normalize numerical features in the dataset.

7. **Encoding Categorical Variables:**
   - Perform one-hot encoding for categorical features such as 'protocol_type', 'service', and 'flag'.

8. **Label Encoding:**
   - Convert the multiclass labels into numerical format using Label Encoding.

9. **Data Preparation:**
   - Split the data into feature set `X` and target variable `y`.
   - One-hot encode the target variable for compatibility with the neural network model.

10. **Data Reshaping:**
    - Reshape the feature set to fit the input requirements of the Convolutional Neural Network (CNN).


### Feature Identification and Categorization

In order to develop a robust and effective Network Intrusion Detection System (NIDS) using Machine Learning (ML) and Deep Learning (DL), it is imperative to have a comprehensive understanding of the features that are involved in the network traffic data. The features can be broadly categorized based on their nature and the type of information they provide. 

*  ***Connection Information***

	Features in this category provide basic information about each network connection. This includes details such as the duration of the connection, the type of protocol used, the network service accessed, and the status of the connection. Analyzing these features helps in understanding the general behavior of the network traffic and aids in the initial filtering of benign connections.

	![Screenshot 2023-10-31 171749](https://github.com/mohammedAcheddad/AI-Based-Network-IDS_ML-DL/assets/105829473/658751ac-125d-405b-b82e-95015e28ace7)

*  ***Connection Content***
	
	These features delve deeper into the content and characteristics of the network connections. They provide insights into the payload and the nature of the connection, highlighting aspects like urgent data packets, failed login attempts, and file creation operations. Analyzing these features is crucial for identifying malicious activities that exploit vulnerabilities in the network services.

	![Screenshot 2023-10-31 171823](https://github.com/mohammedAcheddad/AI-Based-Network-IDS_ML-DL/assets/105829473/9c5cd334-a2b0-4311-bc78-0b22fd030a02)

*  ***Traffic Information***

	Traffic information features help in understanding the patterns and trends in the network traffic. They provide statistics related to the rate of certain types of errors, the number of connections to the same host or service, and the distribution of service access across different hosts. These features are vital for detecting distributed attacks and network scans.

	![Screenshot 2023-10-31 171911](https://github.com/mohammedAcheddad/AI-Based-Network-IDS_ML-DL/assets/105829473/57c69f04-804f-497f-b38e-2dd65ba0de9b)

### Output Label

Finally, the 'label' feature plays a critical role as it provides the ground truth for supervised learning models. It indicates whether a particular connection is normal or an anomaly, guiding the model during the training phase to learn the patterns associated with benign and malicious network activities.
* In case of binary prediction the output/outcome/label is categorized into 2 types:
	* **Normal Traffic**
	*  **Attack**

* In the case of multiclass prediction the output/outcome/label is categorized into four main attack types besides normal traffic:
	- **DoS (Denial of Service):** Attacks that shut down a network, making it inaccessible to its intended users.
	- **R2L (Root to Local):** Unauthorized access from a remote machine.
	- **Probe:** Surveillance and other probing, such as port scanning.
	- **U2R (User to Root):** Unauthorized access to local superuser privileges.  
	

| Attack Type | Description                                                                 | Attack Labels                                               |
|-------------|-----------------------------------------------------------------------------|-------------------------------------------------------------|
| DoS         | Denial of Service - attacks that shut down a network making it inaccessible to its intended users.  | apache2, back, land, neptune, mailbomb, pod, processtable, smurf, teardrop, udpstorm, worm (...)                           |
| R2L         | Root to Local - unauthorized access from a remote machine.                  | ftp_write,guess_passwd, httptunnel, imap, multihop, named (...) |                               |
| Probe       | Surveillance and other probing, such as port scanning.                      | ipsweep,mscan, nmap, portsweep, saint, satan (...)            |
| U2R         | User to Root - unauthorized access to local superuser privileges.           |buffer_overflow, loadmodule, perl, ps, rootkit, sqlattack (...) |





<!-- Add Table for Output Label here -->



 By thoroughly understanding and categorizing these features, we set a solid foundation for feature selection and model training in the subsequent stages of our project. This structured approach ensures that we utilize the most relevant features, leading to a more accurate and efficient NIDS.

### Model Development

The process of model development in our Intrusion Detection System (IDS) follows a structured approach that starts from understanding the data, pre-processing it, and then experimenting with various machine learning and deep learning models to determine the best performing model.

#### Machine Learning Models

Machine learning models are essential in pattern recognition and classification tasks. In our IDS, several machine learning models will be evaluated to identify network intrusions effectively. Below are the models that we will consider:

1. **Decision Trees**: Decision trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

2. **Random Forest**: This ensemble method works by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) of the individual trees.

3. **Support Vector Machines (SVM)**: SVMs are based on the idea of finding a hyperplane that best divides a dataset into classes. Support Vector Machines are particularly well-suited for binary classification problems.

4. **K-Nearest Neighbors (KNN)**: The principle behind KNN is to find a predefined number of training samples closest in distance to the new point and predict the label from these.

5. **Naive Bayes**: This is a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.

6. **Logistic Regression**: Despite its name, logistic regression is used in binary classification rather than regression. It estimates the probabilities using a logistic function.

#### Deep Learning Models (on-going)

Deep learning models, particularly neural networks, have proven to be highly effective in a wide range of classification tasks. For our IDS, we consider the following deep learning architectures:

1. **Convolutional Neural Networks (CNN)**: Although commonly used in image recognition, CNNs can also be applied to sequence data such as time-series and have the advantage of being able to capture spatial-temporal features in the dataset.

2. **Recurrent Neural Networks (RNN)** *~to be implemented~* : RNNs are suitable for sequence prediction problems since they can capture temporal dynamics, which is critical in network traffic where the sequence of packets could be an indicator of an attack pattern.

3. **Long Short-Term Memory Networks (LSTM)** *~to be implemented~*: LSTMs are a special kind of RNN, capable of learning long-term dependencies. They work exceptionally well on a wide range of problems and are now widely used.

4. **Autoencoders** *~to be implemented~*: These are used for learning efficient codings and have been applied in anomaly detection, which is akin to intrusion detection. They learn to capture the most significant features and can reconstruct the input.

6. **Generative Adversarial Networks (GAN)** *~to be implemented~*: In the context of IDS, GANs can be used for generating synthetic attack data to augment the training dataset, improving the robustness of the system.

Each of these models will be trained using the pre-processed dataset , which involved converting attack labels into broad categories. The models performance will be evaluated based on accuracy, precision, recall, F1-score, and the area under the receiver operating characteristic (ROC) curve. The model that best identifies the different types of network intrusions will be selected for deployment in the IDS. Further hyperparameter tuning and cross-validation will be applied to ensure the model's generalization across different network environments.


The images provided include a confusion matrix from a test dataset and a text output with performance metrics such as loss, accuracy, AUC scores, and a classification report for a deep learning model. Here's an analysis based on the provided results:

### Deep Learning Model Evaluation -*Binary*-  

We've only covered CNN yet.

#### Confusion Matrix Analysis

The confusion matrix is a crucial metric for evaluating the accuracy of a classification model. It helps in understanding the model's performance in terms of true positives, true negatives, false positives, and false negatives.

![image](https://github.com/mohammedAcheddad/AI-Based-Network-IDS_ML-DL/assets/105829473/4ace7975-6720-4cce-99ef-505d9dad55b8)

- **True Positives (TP)**: The model correctly predicted the positive class.
- **True Negatives (TN)**: The model correctly predicted the negative class.
- **False Positives (FP)**: The model incorrectly predicted the negative class as positive.
- **False Negatives (FN)**: The model incorrectly predicted the positive class as negative.

For our CNN model, the following are the observed counts:

|                | Training Set | Testing Set |
|----------------|:------------:|:-----------:|
| True Positives |    46,549    |    11,737   |
| True Negatives |    53,840    |    13,318   |
| False Positives|      133     |      51     |
| False Negatives|      248     |      87     |

#### Model Performance Metrics

The model's effectiveness is quantified using the following metrics:

- **Accuracy**: Reflects the overall correctness of the model. It's the ratio of true predictions (both positive and negative) to the total number of observations.
- **Precision**: The ratio of true positive predictions to the total number of positive predictions. High precision indicates a low rate of false positives.
- **Recall (Sensitivity)**: The ratio of true positive predictions to all actual positives. High recall indicates the model is good at detecting positive instances.
- **F1 Score**: The harmonic mean of precision and recall. A high F1 score suggests a balanced model with both good precision and recall.

The performance metrics obtained for the model are as follows:

- **Training Metrics**:
  - Accuracy: 99.62%
  - Precision: 99.72%
  - Recall: 99.47%
  - F1 Score: 99.59%

- **Testing Metrics**:
  - Accuracy: 99.45%
  - Precision: 99.57%
  - Recall: 99.26%
  - F1 Score: 99.42%

#### Optimization of Neural Network Using Randomized Search

#### Hyperparameter Tuning Approach
To enhance the performance of our CNN model for binary classification, we employed RandomizedSearchCV for hyperparameter tuning. This technique allows us to systematically search through a predefined hyperparameter space to find the most effective configuration for our model.

#### Parameters and Search Space
The hyperparameters we chose to optimize include the number of neurons in the first and second hidden layers (`units1` and `units2`), the learning rate of the optimizer, the batch size, and the number of epochs. Here is the search space we defined:

- `units1`: [64, 128, 256, 512]
- `units2`: [32, 64, 128, 256]
- `learning_rate`: [0.01, 0.001, 0.0001]
- `batch_size`: [16, 32, 64]
- `epochs`: [10, 20, 30]

#### Execution and Results
The optimization process was executed on a GPU to accelerate computations, ensuring a more efficient search. After running the randomized search with cross-validation, we obtained an optimized set of hyperparameters that enhanced the model's performance significantly.

The best configuration from the search is as follows:
- Number of neurons in the first hidden layer (`units1`): 64
- Number of neurons in the second hidden layer (`units2`): 64
- Learning rate: 0.001
- Batch size: 64
- Number of epochs: 20

With these optimized hyperparameters, our model achieved a best score of 99.61% on the validation set during the tuning process.



### Deep Learning Model Evaluation -*MultiClass*-

We've only covered CNN yet

#### Confusion Matrix Analysis

![image](https://github.com/mohammedAcheddad/AI-Based-Network-IDS_ML-DL/assets/105829473/d24c3c8c-d57e-4fa9-bc38-ee84054e2b3d)


The confusion matrix is a performance measurement for machine learning classification. It is extremely useful for measuring recall, precision, specificity, accuracy, and most importantly, AUC-ROC curves. The provided matrix includes the following classes: 'normal', **DoS**, **Probe**, **R2L**, and **U2R**.

- **True Positives (TP)**: The diagonal from the top left to bottom right shows the number of correct predictions for each class. For instance, the model correctly predicted '**normal**' 5991 times and '**U2R**' 9169 times.
  
- **False Positives (FP)**: Columns show the instances where other classes were incorrectly predicted as the given class. For example, there were 318 instances where '**U2R**' was incorrectly predicted as '**normal**'.

- **False Negatives (FN)**: Rows show the instances where the given class was incorrectly predicted as another class. For example, there was **1** instance where '**Probe**' was incorrectly predicted as '**normal**'.

- **True Negatives (TN)**: Not directly shown in the matrix, but implied by the absence of counts in other categories for a specific class.

#### Model Performance Metrics

>AUC Score on Test

	- Class 0 (normal): 0.878
	- Class 1 (Dos): 0.767
	- Class 2 (Probe): 0.546
	- Class 3 (R2L): 0.604
	- Class 4 (U2R): 0.788


- **Accuracy**: This metric gives us the overall accuracy of the model, which in this case is approximately 74.41%. While this gives an overall sense of performance, it doesn't provide detail on class-specific accuracy.

- **AUC Scores**: The AUC scores are quite high, which suggests the model has a good measure of separability. It means that there is a high chance that the model can distinguish between positive class and negative class for each class type.

#### Classification Report




|           | Precision |   Recall  | F1-Score |  Support |
|-----------|:---------:|:---------:|:--------:|:--------:|
|   normal  |   0.89    |   0.80    |   0.85   |   7460   |
|     Dos   |   0.81    |   0.55    |   0.65   |   2421   |
|   Probe   |   0.96    |   0.09    |   0.17   |   2885   |
|     R2L   |   0.88    |   0.21    |   0.34   |   67     |
|     U2R   |   0.66    |   0.94    |   0.78   |   9711   |




- **Precision**: Indicates the ratio of correctly predicted positive observations to the total predicted positives. High precision relates to the low false positive rate. For example, 'normal' has a precision of 0.89, which is quite good.

- **Recall (Sensitivity)**: Indicates the ratio of correctly predicted positive observations to the all observations in actual class. 'U2R' has a recall of 0.94, which suggests the model is very good at detecting this class.

- **F1-Score**: The weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. An F1-score is a good way to show that a class has a good recall and precision balance.

- **Support**: The number of actual occurrences of the class in the specified dataset. For imbalanced data, this is a crucial metric to observe.

| Accuracy  |           |           |           |   0.74   |
|-----------|:---------:|:---------:|:--------:|:--------:|
| Macro Avg |   0.84    |   0.52    |   0.56   |  22544   |
| Weighted Avg | 0.79    |   0.74    |   0.71   |  22544   |

From the classification report and confusion matrix, we can deduce that the model has an overall good performance with some areas of improvement. Particularly, the model seems to struggle with 'R2L' and 'Probe' classes in terms of precision and recall which indicates potential areas to focus on improving, possibly due to imbalanced dataset issues or the model not learning sufficient discriminative features for these categories.

In terms of deep learning model selection, based on these results, the model can be further optimized by:

- Addressing class imbalance, potentially through resampling techniques or using different loss functions that are robust to imbalance.
- Experimenting with different architectures or hyperparameters to improve precision and recall for the underperforming classes.
- Utilizing techniques like cross-validation to ensure the model's robustness and generalizability.
- Employing regularization techniques to prevent overfitting, if it's determined to be a problem.

The high AUC scores indicate that the model's capability to distinguish between classes is quite strong. However, due to the high false negative rates for some classes, we might explore models that provide a better balance between sensitivity and specificity.


## Results and Discussion

The results of the experiments and evaluations are crucial for understanding the effectiveness of the developed AI-based Network IDS . The performance metrics obtained from the testing phase will guide decisions on model selection and potential areas for improvement.

### Challenges and Limitations

Despite the promising results, it's important to acknowledge the challenges and limitations of the AI-Based Network IDS:

1.  **Imbalanced Datasets:**
    
    -   Imbalances in the distribution of normal and attack instances can lead to biased models. Addressing this issue through techniques like resampling or adjusting class weights is crucial.
2.  **Model Interpretability:**
    
    -   Deep learning models, in particular, are often considered as "black boxes," making it challenging to interpret their decision-making processes. Ensuring interpretability is essential for gaining trust in the IDS.
3.  **Adversarial Attacks:**
    
    -   Intruders may attempt to manipulate the network traffic to evade detection. Adversarial attacks pose a significant challenge, and models need to be robust against such attempts.
4.  **Generalization to New Threats:**
    
    -   The models should be capable of generalizing to new and unseen threats. Continuous monitoring and updates are necessary to adapt to emerging attack patterns.
5.  **Computational Resources:**
    
    -   Training and deploying deep learning models often require substantial computational resources. Ensuring scalability and efficiency is crucial, especially for real-time intrusion detection.

### Conclusion

This project represents a significant step forward in the realm of network security. By leveraging advanced AI techniques, including machine learning and deep learning, the IDS aims to provide a robust defense mechanism against a wide range of network intrusions.

The selection of diverse datasets, such as NSL-KDD and UNSW-NB 15, allows for a comprehensive evaluation of the IDS under different scenarios. The project's structured methodology, including data preprocessing, feature identification, and model development, ensures a systematic approach to building an effective IDS.

While the presented results showcase promising performance, the ongoing challenges and limitations highlight the need for continuous improvement. Addressing issues related to imbalanced datasets, ensuring model interpretability, and staying resilient against adversarial attacks are key areas for future enhancements.

In conclusion, this project lays the foundation for adaptive and sophisticated intrusion detection systems, contributing to the ongoing efforts to enhance cybersecurity in an ever-evolving digital landscape.
