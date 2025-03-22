

COMMENTS INDECENCY DETECTOR

Yashika Singh, Yashaswi Rajesh Patki, Hang Yu, & Praveen Kumar Govind ReddyYeshiva University

Abstract
The proliferation of harmful and toxic speech on social media has escalated concerns about online safety and the quality of digital discourse (Namdari, 2023). This study addresses the urgent need for effective tools to identify and manage such content. We propose a multi-label classification model designed to detect various forms of harmful speech in social media comments, including toxic, severely toxic, obscene, threatening, insulting, and identity-related harm (Zhang, Zhao, & LeCun, 2015). Utilizing natural language processing techniques, our model leverages logistic regression extended through the MultiOutputClassifier framework. The study demonstrates the model's effectiveness in classifying diverse harmful behaviors and discusses its limitations and potential improvements. This work contributes to enhancing online user safety and mitigating the impact of harmful speech.

Introduction
The rise of social media has transformed communication, but it has also led to an increase in harmful and toxic content (ElSherief, Kulkarni, & Wang, 2020). This trend has sparked significant concern among researchers, policymakers, and platform administrators about how to effectively manage and mitigate the spread of such content. The challenge lies in developing robust systems that can accurately detect and classify various types of harmful speech in real time. Addressing this issue is crucial for improving online interactions and ensuring user safety.

This study aims to develop and evaluate a multi-label text classification model that identifies different types of harmful comments on social media platforms. By focusing on categories such as toxic, severely toxic, obscene, threatening, insulting, and identity-related harm, the research combines advanced text preprocessing, feature extraction techniques, and logistic regression to create a comprehensive solution. The use of the MultiOutputClassifier framework and GridSearchCV for hyperparameter optimization enhances the model's performance (Curry, Abercrombie, & Rieser, 2021). This research contributes to the broader effort of refining online content moderation systems and fostering a safer digital environment.

Background and Literature Review
Previous Work on Content Moderation

Content review has evolved from an early manual process to using complex digital technology to solve problems (Kennedy, Bacon, Sahn, & von Vacano, 2020). Initially, content review relied on rule-based systems that used keyword matching to filter out inappropriate content. Although these methods are simple and effective, they are difficult to cope with the complexity and subtle differences of natural language, resulting in frequent false positives and omissions of harmful content.

With the advancement of machine learning, automatic content review systems have emerged on the market, utilizing supervised learning techniques to improve accuracy. Early technicians used models that utilized traditional classifiers such as Support Vector Machines (SVM) and Naive Bayes to detect harmful content based on features extracted from text. The introduction of deep learning methods, such as Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), further enhances these systems by capturing finer patterns and contextual information.

In recent years, multi-label classification has become the key to solving the complexity of modern content review. This method allows for the simultaneous detection of various types of harmful behaviors, such as toxicity, obscenity, and threats. MultiOutputClassifier and other technologies achieve a more comprehensive evaluation by predicting multiple labels for each content. Despite these advances, there are still ongoing challenges in adapting to the constantly evolving language and balancing content regulation with freedom of speech (ElSherief, Kulkarni, & Wang, 2020).

Limitations of Existing Models

Despite the advances in content moderation models, several limitations hinder their effectiveness and reliability. First, many traditional models, including those based on simple keyword filtering or rule-based systems, lack the ability to understand context and nuance in language. This often leads to high false positive and false negative rates, where benign content may be flagged as harmful, and harmful content may go undetected.

Second, the reliance on single-label classification models can be restrictive in a multi-label environment like content moderation, where a single piece of content can exhibit multiple types of harmful behavior simultaneously (Zhang, Zhao, & LeCun, 2015). Single-label models typically treat each type of harmful behavior independently, ignoring potential correlations between them. This can result in suboptimal performance, as the interdependencies between different types of harmful content are not leveraged to improve detection accuracy.

Furthermore, existing models often struggle with handling the vast and ever-evolving variety of offensive language on the internet. Adversarial users continuously invent new slang, euphemisms, and obfuscations to bypass moderation systems. Traditional models, which may rely heavily on predefined vocabularies and fixed patterns, can quickly become outdated and ineffective.

Methodology

Dataset

The dataset utilized in this study consists of a comprehensive collection of comments from multiple sources. The following datasets were used:

Training Set:

YouTube Comments: 1,000 hand-labeled comments categorized into various toxicity types (Namdari, 2023).

Wikipedia Talk Pages: Comments providing additional examples of toxicity (Kaggle, 2018).

Facebook Dataset: 4,185 labeled comments for abuse, severity, and other dimensions (ElSherief, Kulkarni, & Wang, 2020).

Test Set:

Multi-platform comment hybrid dataset: 135,556 annotated social media comments from Twitter, Reddit, and YouTube (Kennedy, Bacon, Sahn, & von Vacano, 2020).

Data Preprocessing

Data preprocessing is a crucial step in building an effective text classification model. The process includes:

Text Cleaning: Removal of HTML tags, URLs, user mentions, hashtags, and punctuation.

Text Normalization: Lowercasing and removal of non-alphabetic characters.

Emoji Handling: Converting emojis to descriptive text.

Tokenization and Lemmatization: Using the NLTK library for text segmentation and base form conversion.

Feature Extraction: Transforming text into TF-IDF features with a maximum of 10,000 features.

Model Architecture

We utilized logistic regression as the base classifier combined with MultiOutputClassifier for multi-label classification. Hyperparameters were optimized using GridSearchCV with values for the regularization parameter C (0.01, 0.1, 1, 10, 100).

Training and Evaluation

The model's training and evaluation followed these steps:

Data splitting (80:20 ratio for training and validation sets).

Training using GridSearchCV for optimal parameters.

Evaluation using ROC AUC, accuracy, precision, recall, and F1 score.

Results

Our model achieved the following performance metrics:

ROC AUC: 0.9745

Accuracy: 0.9101

Precision: 0.7084

Recall: 0.4107

F1 Score: 0.5026

Discussion

Strengths:

High ROC AUC (0.9745) and accuracy (0.9101) demonstrate strong overall performance.

Multi-label classification effectively identifies multiple harmful content categories.

Challenges and Future Directions:

Improve recall using advanced models (e.g., transformer-based architectures).

Address performance variations across different comment lengths.

Expand datasets for greater model generalization.

Conclusion

This study developed a robust multi-label classification model to detect various forms of harmful speech, contributing to improving online content moderation and user safety.

References

Namdari, R. (2023). YouTube toxicity data [Data set]. Kaggle.Zhang, L., Zhao, J., & LeCun, Y. (2015).

