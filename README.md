# Suicide-Prevention
This project addresses a major social issue that is mostly overlooked in society and we fail collectively as humans. Our main aim is to spread awareness about mental health
 issues in vulnerable age groups.

 <p align="center" style="font-family:cursive;"> <i><q>This life. This night.<br> Your story. Your pain. <br>Your hope. It matters. <br>It all matters.</q></i><br> -Jamie Tworkowski</p>
 
 ##### Talk Suicide Canada: 1-833-456-4566 (Phone) | 45645 (text, 4 p.m. to midnight ET only) [talksuicide.ca/parlonssuicide.ca](talksuicide.ca/parlonssuicide.ca)

 ## Method
In this project, we perform an analysis of the suicide dataset from various demographics available on Kaggle.<br><br> We create a projection(Regression) model that projects the possible suicide rate in the following year given the parameters like Country, Year, Suicide_Rate, Gender_Identity, Human Development Index, GDP, GDP_per_capita, Population, and Generation(Age-groups) involved.<br>
<br>We further predict the most vulnerable age group for a given country. This is a classification problem as we classify this age-group as most vulnerable and needs to be looked after.

## Statistical Analysis
A statistical study is done to choose the best model over a number of sample test cases. The picture below shows the ROC curve of the methods tested. We can see that Random Forest has the highest True positive and least False positive rate.
<p align="center">
    <img src="https://github.com/imjunaida/Suicide-Prevention/blob/main/ROC_DecisionTree.png" width= 250 height= auto>
    <img src="https://github.com/imjunaida/Suicide-Prevention/blob/main/ROC_knn.png" width= 250 height= auto>
    <img src="https://github.com/imjunaida/Suicide-Prevention/blob/main/ROC_Random Forest.png" width= 250 height= auto>  
</p>
<p align ="center"> <em>Decision Tree &emsp;&emsp;&emsp;&emsp;&emsp;   &emsp;&emsp;&emsp;&emsp; KNN Classifier &emsp;&emsp;&emsp;&emsp;   &emsp;&emsp;&emsp;&emsp;&emsp; Random Forest</em></p>
