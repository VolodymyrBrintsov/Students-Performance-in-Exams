import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, chi2_contingency, ttest_1samp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

students_performance = pd.read_csv('data/StudentsPerformance.csv')
pd.set_option('display.max_columns', None)

#Inspecting first 5 rows of the data
print(students_performance.head(), '\n')
scores = ['math score', 'reading score', 'writing score']

#Dataframe info
print(students_performance.info(), '\n')

#Proportions of specific categorical data
gender_proportion = students_performance["gender"].value_counts().reset_index()
race_proportion = students_performance["race/ethnicity"].value_counts().reset_index()
preparation_proportion = students_performance["test preparation course"].value_counts().reset_index()
lunch_proportion = students_performance["lunch"].value_counts().reset_index()
parental_education_proportion = students_performance["parental level of education"].value_counts().reset_index()
proportions = [gender_proportion, race_proportion, preparation_proportion, lunch_proportion, parental_education_proportion]

plt.figure(figsize=(12, 6))
for i, proportion in enumerate(proportions):
    plt.subplot(1, 5, i+1)
    plt.title(f'{proportion.columns[-1]} proportion')
    plt.pie(proportion.iloc[:, 1], autopct='%d%%')
    plt.axis('equal')
    plt.legend(proportion.iloc[:, 0])
plt.tight_layout()
plt.show()

#Visualize math, reading and writing score depending on ethnitical group, gender, test preparation courses, parental level of education and lunch
dependencies = ['race/ethnicity', 'gender', 'test preparation course', 'parental level of education', 'lunch']
for result in scores:
    plt.figure(figsize=(18, 9))
    for count, dependence in enumerate(dependencies):
        plt.subplot(1, 5, count+1)
        sns.barplot(y=result, data=students_performance, x=dependence)
        plt.title(f'{result} vs {dependence}')
        plt.xticks(rotation=-15)
    plt.tight_layout()
    plt.show()

#Visualise distribution of scores dependent on gender, test preparation and ethnitical groups
dependencies = ['race/ethnicity', 'gender', 'test preparation course']
for result in scores:
    plt.figure(figsize=(18, 9))
    for count, dependence in enumerate(dependencies):
        plt.subplot(1, 3, count + 1)
        sns.kdeplot(result, data=students_performance, hue=dependence, shade=True)
        plt.title(f'{result} vs {dependence}')
        sns.despine()
    plt.tight_layout()
    plt.show()

#
completed_preparation = students_performance[students_performance['test preparation course'] == 'completed']
none_preparation = students_performance[students_performance['test preparation course'] == 'none']

male_performance = students_performance[students_performance['gender'] == 'male']
female_performance = students_performance[students_performance['gender'] == 'female']

groupA_performance = students_performance[students_performance['race/ethnicity'] == 'group A']
groupB_performance = students_performance[students_performance['race/ethnicity'] == 'group B']
groupC_performance = students_performance[students_performance['race/ethnicity'] == 'group C']
groupD_performance = students_performance[students_performance['race/ethnicity'] == 'group D']
groupE_performance = students_performance[students_performance['race/ethnicity'] == 'group E']

#Check difference in standard deviation in scores of gender
male_scores_std = np.std(male_performance)
female_scores_std = np.std(female_performance)
difference_scores_std_gender = abs(male_scores_std - female_scores_std)
print('Difference of standard deviation of genders:\n', difference_scores_std_gender, '\n') ## Standart deviation of scores are about to be the same

#Check difference in standard deviation in scores of courses
completed_preparation_scores_std = np.std(completed_preparation)
none_preparation_scores_std = np.std(none_preparation)
difference_scores_std_courses = abs(completed_preparation_scores_std - none_preparation_scores_std)
print('Difference of standard deviation of courses:\n', difference_scores_std_courses, '\n') ##Standard deviation of scores are about to be the same except writing scores (1.63 difference)

#Check difference in standard deviation in scores of race groups
groupA_scores_std = np.std(groupA_performance)
groupB_scores_std = np.std(groupB_performance)
groupC_scores_std = np.std(groupC_performance)
groupD_scores_std = np.std(groupD_performance)
groupE_scores_std = np.std(groupE_performance)
difference_scores_std_race = abs(groupE_scores_std - groupA_scores_std - groupD_scores_std - groupC_scores_std - groupB_scores_std)
print('Difference of standard deviation of races:\n', difference_scores_std_race, '\n') #There is a big difference in spread, so we can not check hypothesis on this data

#Testing association between test preparation and scores
completed_preparation = students_performance[students_performance['test preparation course'] == 'completed']
none_preparation = students_performance[students_performance['test preparation course'] == 'none']

#Math score and test prep
tstat, test_prep_pval = ttest_ind(completed_preparation[scores], none_preparation[scores])
print('''
H0 - mean score of students who completed course and not are the same.
HA - mean score of students who completed course and not are not the same.
''')
for score, result in enumerate(test_prep_pval):
    if result < 0.05:
        print(f'For {scores[score]} we accept that they have different mean score.')
    else:
        print(f'For {scores[score]} we accept that they have the same mean score.')

### THERE IS A BIG DIFFERENCE IN MEAN SCORES OF STUDENT WHO COMPLETED EXTRA COURSES AND NOT ###
#Inspecting the mean values of student who completed course and not completed
completed_preparation_scores_mean = np.mean(completed_preparation)
none_preparation_scores_mean = np.mean(none_preparation)
print(f'Mean for students who completed courses\n{completed_preparation_scores_mean}\nMean for those who not completed courses\n{none_preparation_scores_mean}')
#In average those students who completed courses has more scores than those who does not complete courses

#Check association between gender and test results
#Math score and test gender
tstat, gender_scores_pval = ttest_ind(male_performance[scores], female_performance[scores])
print('''
H0 - mean score of students of both genders are the same.
HA - mean score of students of both genders are not the same.
''')
for score, result in enumerate(gender_scores_pval):
    if result < 0.05:
        print(f'For {scores[score]} we accept that both genders have different mean score.')
    else:
        print(f'For {scores[score]} we accept that both genders have the same mean score.')

### BOTH GENDERS HAVE DIFFERENT SCORES
#Inspecting mean value of scores by genders
male_scores_mean = np.mean(male_performance)
female_scores_mean = np.mean(female_performance)
print(f'Mean of scores for male \n{completed_preparation_scores_mean}\nMean of scores for female \n{none_preparation_scores_mean}')
#In average male student have more math score but have less reading and writing score

#Cross tabulation of race/ethnicity and parental level of education
race_education_tab = pd.crosstab(students_performance['race/ethnicity'], students_performance['parental level of education'])
#Association between race/ethnicity and parental level of education
chi2, race_education_pval, dof, exp = chi2_contingency(race_education_tab)
print('''
H0 - there is an association parental level of education and ethnicity group.
HA - there is no association between parental level of education and writing ethnicity group.
''')
result = 'Reject H0' if race_education_pval < 0.05 else "Accept H0"
print(f"Chi2: {chi2}, confident level: {race_education_pval}\n")
print(result) ## There is an association in this parameters

#What is the most influenced parameters for scores
plt.figure(figsize=(18, 9))
for i, score in enumerate(scores):
    feature_set = students_performance.drop(scores, axis=1)
    for column in feature_set.columns:
        encoder = LabelEncoder()
        feature_set[column] = encoder.fit_transform(feature_set[column])
    target = students_performance[score]
    linear_model = LinearRegression().fit(feature_set, target)
    ax = plt.subplot(1, 3, i+1)
    coefficients = abs(linear_model.coef_)
    x_values = range(len(coefficients))
    plt.bar(x_values, coefficients)
    plt.title(f'Influence of Parameter by Linear Regression for {score}')
    plt.ylabel('Coefficients')
    plt.xlabel('Parameters')
    plt.xticks(rotation=-15)
    ax.set_xticks(x_values)
    ax.set_xticklabels(feature_set.columns)
plt.tight_layout()
plt.show()


