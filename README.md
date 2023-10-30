


## Dataset Choice: Embracing Diversity with NSL-KDD and UNSW-NB 15

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
