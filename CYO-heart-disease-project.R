# title: "CYO project: Heart Disease prediction"
# version: "v1.2"
# author: "Kiko Núñez"
# date: "2020/01/06"
# 
# 0. Load needed libraries ####
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(tidyverse)  # For data cleaning, sorting, and visualization

if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(knitr) # For tidy tables

if(!require(DataExplorer)) install.packages("DataExplorer", repos = "http://cran.us.r-project.org")
library(DataExplorer) # For Exploratory Data Analysis

if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
library(gridExtra) # To plot several plots in one figure

if(!require(ggpubr)) install.packages("ggpubr", repos = "http://cran.us.r-project.org")
library(ggpubr) # To prepare publication-ready plots

if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
library(GGally) # For correlations

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret) # For training models

if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
library(e1071) # For training models

if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
library(glmnet) # For training models

if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
library(rpart) # For classification model

# ## 1. Introduction ####
# 
# This project aims at putting into practice the contents learned during the Data Science course making use of the publicly available dataset [Heart Disease UCI | Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci) and then applying some of the tools and methods explained during the course. In addition, the project explores the use of other R libraries beyond those addressed during the course.
# 
# ### 1.1 Dataset description
# 
# As explained in the Kaggle [documentation](https://www.kaggle.com/ronitf/heart-disease-uci) of this dataset, the Heart Disease UCI dataset contains 76 attributes, but all published experiments refer to using a subset of 14 of them:
# 
# 1. *age*: age in years.
# 2. *sex*: (1 = male; 0 = female).
# 3. *cp*: chest pain type  (typical angina, atypical angina, non-angina, or asymptomatic angina).
# 4. *trestbps*: resting blood pressure (in mm/Hg on admission to the hospital).
# 5. *chol*: serum cholestoral in mg/dl.
# 6. *fbs*: Fasting blood sugar (<120 mg/dl or >120 mg/dl) (1 = true; 0 = false).
# 7. *restecg*: resting electrocardiographic results (normal, ST-T wave abnormality, or left ventricular hypertrophy).
# 8. *thalach*: Max. heart rate achieved during thalium stress test.
# 9. *exang*: Exercise induced angina (1 = yes; 0 = no).
# 10. *oldpeak*: ST depression induced by exercise relative to rest.
# 11. *slope*: Slope of peak exercise ST segment (0 = upsloping, 1 = flat, or 2 = downsloping).
# 12. *ca*: number of major vessels (0-3, 4 = NA) colored by flouroscopy.
# 13. *thal*: Thalium stress test result; 3 = normal; 6 = fixed defect; 7 = reversable defect; 0 = NA.
# 14. *target*: Heart disease status 1 or 0 (0 = heart disease 1 = asymptomatic).  
# 
# ### 1.2 Goal of the project
# 
# The main goal of this project is to apply the insights gained during the course to predict whether a subject has (or not) a heart disease depending on a set of related variables.  
# 
# ### 1.3 Key steps
# 
# 1. Data analysis and visualization. We'll make a comprehensive analysis of the variables distribution in the overall dataset and also per gender and health status.
# 2. Partition into train and test datasets. Before training the models, we'll leave the 20% of the samples out for validation purposes.
# 3. Train the models. We'll train 3 different models to compare their accuracy: Random Forest (RF), Decision Trees (DT) and General Linear Model (GLM).
# 4. Apply trained models to test dataset.
# 5. Conclusions, limitations and future work.
# 
# ## 2. Data Analysis and Visualization ####
# 
# ### 2.1. Initial Analysis
# 
# The first step is to load the dataset, transform it into a tidy format, and then explore it:

df <- read_csv("./heart.csv") # To read local file
copy <- df # Copy for use later
df2 <- df %>% filter(thal != 0 & ca != 4) %>%  # remove values corresponding to NA in original dataset
# Recode the categorical variables as factors using the dplyr library:
  mutate(
    sex = case_when(
      sex == 0 ~ "female",
      sex == 1 ~ "male"
           ),
    fbs = case_when(
      fbs == 0 ~ "<=120",
      fbs == 1 ~ ">120"
            ),
    exang = case_when(
      exang == 0 ~ "no",
      exang == 1 ~ "yes"
            ),
    cp = case_when(
      cp == 3 ~ "typical angina",
      cp == 1 ~ "atypical angina",
      cp == 2 ~ "non-anginal",
      cp == 0 ~ "asymptomatic angina"
          ),
    restecg = case_when(
      restecg == 0 ~ "hypertrophy",
      restecg == 1 ~ "normal",
      restecg == 2 ~ "wave abnormality"
              ),
    target = case_when(
      target == 1 ~ "asymptomatic",
      target == 0 ~ "heart-disease"
              ),
    slope = case_when(
      slope == 2 ~ "upsloping",
      slope == 1 ~ "flat",
      slope == 0 ~ "downsloping"
    ),
    thal = case_when(
      thal == 1 ~ "fixed defect",
      thal == 2 ~ "normal",
      thal == 3 ~ "reversable defect"
    ),
    sex = as.factor(sex),
    fbs = as.factor(fbs),
    exang = as.factor(exang),
    cp = as.factor(cp),
    slope = as.factor(slope),
    ca = as.factor(ca),
    thal = as.factor(thal),
    restecg = as.factor(restecg),
    target = as.factor(target)
  )

kable(head(df2[,1:7]), caption = "Head of the dataset. Columns 1 to 7.") # Check that the transformnation worked
kable(head(df2[,8:14]), caption = "Head of the dataset. Columns 8 to 14.") # We split into 2 tables for visualization purposes.
df <- df2 # Replace the df dataset by the tidy dataset


# We notice that we have the 14 variables as explained in the introduction section.
# 
# ### 2.2 Data visualization  
# 
# In this section, we'll apply some functions to the entire dataset in order to gain more insights about it through its visualization.
# 
# #### 2.2.1 Visualize the data summary and distribution for each variable.
# A dataset summary can be provided by the `summary()` function:

df %>% summary()

# Now we'll use some functions from the `DataExplorer` library to visualize the distribution of the continuous and categorical variables:

plot_density(df, ggtheme = theme_minimal(), 
             title = "Distribution of continuous variables", 
             geom_density_args = list("fill" = "grey", "alpha" = 0.5))
plot_bar(df, ggtheme = theme_minimal(),
         title = "Distribution of categorical variables")

# Here we can extract the following information:  
# 
# * Age distribution is slightly skewed towards the odest values, being the mean about 55 years old..
# * Cholesterol levels are normally distributed around 250, with some outliers in the upper levels.
# * Most of ST depression tests are between 2 and 4 mm.
# * Maximum heart rate distribution is slightly skewed towards the upper side, being the mean around 160 bpm.
# * Resting blood pressure is slgihtly skewed towards the lower end around 130 mm/Hg.
# * Regarding gender, there are two times more males than females in the dataset.
# * Most of the cases reported asymptomatic angina when asked about chest pain.
# * Fasting blood sugar was mostly rated under 120 mg/dl.
# * Resting EKG results were almost equally distributed between normal and hypertorphy findings.
# * Exercise induced angina was found in 1/3 of the cases included in the dataset.
# * Slope of peak exercise ST segment was reported mostly on upsloping and flat in an equal basis.
# * Number of major vessels colored in the fluoroscopy was found to be 0 for most of the subjects.
# * Thalium stress test resulted in normal or reversable defect for most of the cases.
# * The dataset is a bit biased towards asymptomatics subjects compared to heart disease subjects.  
# 
# 
# #### 2.2.2 Continuous variables distribution per gender. 
# In this section, we are going to analyze more in depth the continuous variables distributions per gender:

# Male and Female count
a1 <- ggplot(df, aes(x = sex, fill = sex)) +
  geom_bar(width = 0.5) + 
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none') +
  labs(x = "", title = "Gender")

# Age per gender
b1 <- ggplot(df, aes(x= sex, y = age, fill = sex)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) +
  ylim(0, 90) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none') +
  labs(x = "", y = "Years old", title = "Age distribution")

# trestbps
c1 <- ggplot(df, aes(x = sex, y = trestbps, fill = sex)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "mmHg", title = "Blood pressure distribution") +
  ylim(0,250) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# chol
d1 <- ggplot(df, aes(x = sex, y = chol, fill = sex)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "mg/dl", title = "Cholesterol distribution") +
  ylim(0,500) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# oldpeak
e1 <- ggplot(df, aes(x = sex, y = oldpeak, fill = sex)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "mm", title = "ST depression distribution") +
  ylim(0,10) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# thalach
f1 <- ggplot(df, aes(x = sex, y = thalach, fill = sex)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "bpm", title = "Max. heart rate distribution") +
  ylim(0,250) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

ggarrange(a1, b1, c1, d1, e1, f1, 
          ncol = 2, nrow = 3, common.legend = TRUE,
            align = "hv") # It may take a while to print this plot

# With a first visual inspection, in these plots we notice that the datset has two times more males than females and that females have a slightly higher maximum heart beat rate than males. The other variables are more or less equally distributed in both groups.

# #### 2.2.3 Categorical variables distribution per gender.
# In this section, we'll perform a similar analysis on the categorical variables.  

# Disease status
g1 <- ggplot(df, aes(x = target, fill = sex)) +
  geom_bar(width = 0.5, position = 'dodge') + 
  labs(x = "", y = "", title = "Target") +
  coord_flip() +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# cp
h1 <- ggplot(df, aes(cp, group = sex, fill = sex)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Chest pain") +
  coord_flip() +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# restecg
i1 <- ggplot(df, aes(restecg, group = sex, fill = sex)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Rest. EKG") +
  coord_flip() +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# slope
j1 <- ggplot(df, aes(slope, group = sex, fill = sex)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Peak exercise ST") +
  coord_flip() +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# thal 
k1 <- ggplot(df, aes(thal, group = sex, fill = sex)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Thalium stress test") +
  coord_flip() +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# fbp
l1 <- ggplot(df, aes(fbs, group = sex, fill = sex)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Fasting blood sugar") +
  coord_flip() +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# exang
m1 <- ggplot(df, aes(exang, group = sex, fill = sex)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Exercise induced angina") +
  coord_flip() +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

# ca
n1 <- ggplot(df, aes(ca, group = sex, fill = sex)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Flouroscopy") +
  coord_flip() +
  scale_fill_manual(values = c("pink","cyan"))+
  theme_minimal() +
  theme(legend.position='none')

ggarrange(g1, h1, i1, j1, k1, l1, m1, n1, 
          ncol = 2, nrow = 4, common.legend = TRUE,
          align = "hv") # It may take a while to print this plot


# From this set of plots, taking into account that the sample of males is twice the sample of females, we can notice the following differences in the distributions:  
# 
# * The dataset is biased towards asymptomatic population, with a higher proportion of males belonging to the heart disease group.
# * The Thalium stress test resulted in reversable defect in a higher proportion of males.
# * Wave abnormalities in the resting EKG were found in a higher proportion of females.
# * Males were more likely to suffer exercise induced angina than females.
# 
# 
# #### 2.2.4 Continuous variables distribution per gender and disease status.
# Now let's perform a similar visual analysis but in this case let's put the focus on the disease status instead. We'll also make different plots considering the gender of the patients.
# 
# **Male subjects:**  
df <- df2 %>% filter(sex == "male")

# Disease status count
a2 <- ggplot(df, aes(x = target, fill = target)) +
  geom_bar(width = 0.5, position = 'dodge') + 
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none') +
  labs(x = "", y = "", title = "Target")

# Age per gender
b2 <- ggplot(df, aes(x= target, y = age, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) +
  ylim(0, 90) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none') +
  labs(x = "", y = "Years old", title = "Age distribution")

# trestbps
c2 <- ggplot(df, aes(x = target, y = trestbps, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) +
  labs(x = "", y = "mmHg", title = "Blood pressure distribution") +
  ylim(0,250) +
 stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# chol
d2 <- ggplot(df, aes(x = target, y = chol, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "mg/dl", title = "Cholestorol distribution") +
  ylim(0,500) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# oldpeak
e2 <- ggplot(df, aes(x = target, y = oldpeak, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "mm", title = "ST depression distribution") +
  ylim(0,10) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# thalach
f2 <- ggplot(df, aes(x = target, y = thalach, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "bpm", title = "Max. heart rate") +
  ylim(0,250) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

ggarrange(a2, b2, c2, d2, e2, f2, 
          ncol = 2, nrow = 3, common.legend = TRUE,
            align = "hv") # It may take a while to print this plot

# Male patients with heart disease are significantly older, have higher cholesterol level, higher ST segment depression and lower maximum heart rate response to the Thallium test.  
# 
# Now let's analyze the categorical variables for males:  
# Disease status
g2 <- ggplot(df, aes(x = target, fill = target)) +
  geom_bar(width = 0.5, position = 'dodge') + 
  labs(x = "", y = "", title = "Target") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# cp
h2 <- ggplot(df, aes(cp, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Chest pain") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# restecg
i2 <- ggplot(df, aes(restecg, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Rest. EKG") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# slope
j2 <- ggplot(df, aes(slope, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Peak exercise ST") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# thal 
k2 <- ggplot(df, aes(thal, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Thalium stress test") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# fbp
l2 <- ggplot(df, aes(fbs, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Fasting blood sugar") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# exang
m2 <- ggplot(df, aes(exang, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Exercise induced angina") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# ca
n2 <- ggplot(df, aes(ca, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Flouroscopy") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

ggarrange(g2, h2, i2, j2, k2, l2, m2, n2, 
          ncol = 2, nrow = 4, common.legend = TRUE,
          align = "hv") # It may take a while to print this plot

# From the categorical variables, we could extract the following information:  
# 
# * The dataset has more heart disease males than asymptomatics.
# * Chest pain in males revealed as asymptomatic angina for most of heart disease group.
# * Hypertrophy was found in the resting EKG more often in the heart disease group of males.
# * ST segment during peak exercise showed to be upsloping in most of asymptomatics males, while being flat for most of the subjects belonging to the heart disease group.
# * Thallium stress test resulted in normal for most asymptomatics males while it showed reversable defects in the heart disease group.
# * Fasting blood sugar was mostly under 120 for the males belonging to the heart disease group.
# * Asymptomatic males were more likely to not to show an exercise induced angina compared to heart disease subjects.
# * Fluoroscopy resulted positive for most of the males belonging to the heart disease group.  
# 
#   
# **Female subjects:**  
# 
# Now let's repeat the same analysis with the females group only.
df <- df2 %>% filter(sex == "female")

# Disease status count
a2 <- ggplot(df, aes(x = target, fill = target)) +
  geom_bar(width = 0.5, position = 'dodge') + 
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none') +
  labs(x = "", y = "", title = "Target")

# Age per gender
b2 <- ggplot(df, aes(x= target, y = age, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) +
  ylim(0, 90) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none') +
  labs(x = "", y = "Years old", title = "Age distribution")

# trestbps
c2 <- ggplot(df, aes(x = target, y = trestbps, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) +
  labs(x = "", y = "mmHg", title = "Blood pressure distribution") +
  ylim(0,250) +
 stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# chol
d2 <- ggplot(df, aes(x = target, y = chol, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "mg/dl", title = "Cholestorol distribution") +
  ylim(0,500) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# oldpeak
e2 <- ggplot(df, aes(x = target, y = oldpeak, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "mm", title = "ST depression distribution") +
  ylim(0,10) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# thalach
f2 <- ggplot(df, aes(x = target, y = thalach, fill = target)) +
  geom_violin(width = 0.5, alpha = 0.5) +
  geom_boxplot(width = 0.2) + 
  labs(x = "", y = "bpm", title = "Max. heart rate") +
  ylim(0,250) +
  stat_compare_means(aes(label = ..p.signif..), method = "t.test") +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

ggarrange(a2, b2, c2, d2, e2, f2, 
          ncol = 2, nrow = 3, common.legend = TRUE,
            align = "hv") # It may take a while to print this plot.

  
# From the categorical variables in females, we could extract the following information:  
# 
# * The dataset has less heart disease females than asymptomatics.
# * Chest pain revealed as asymptomatic angina for most of females belonging to the heart disease group, similarly to males.
# * Hypertrophy was found in the resting EKG for females more often in the heart disease group, while most asymptomatics reported a normal EKG.
# * ST segment during peak exercise showed to be upsloping in most of asymptomatics, while being flat for most of the subjects belonging to the heart disease group, similarly to males.
# * Thallium stress test resulted in normal for most asymptomatics femlaes while it showed reversable defects in the heart disease group, similarly to males.
# * Fasting blood sugar was slightly higher for the heart disease subjects.
# * Proportion of asymptomatic and heart disease females reporting fasting blood sugar under 120 is in keeping with the distribution of females.
# * Fluoroscopy resulted positive for most of the subjects belonging to the heart disease group, similarly to males.  
# 
# 
# The following plots show the distributions of the categorical variables for females grouped by target (asymptomatic vs heart disease):  

# Disease status
g2 <- ggplot(df, aes(x = target, fill = target)) +
  geom_bar(width = 0.5, position = 'dodge') + 
  labs(x = "", y = "", title = "Target") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# cp
h2 <- ggplot(df, aes(cp, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Chest pain") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# restecg
i2 <- ggplot(df, aes(restecg, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Rest. EKG") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# slope
j2 <- ggplot(df, aes(slope, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Peak exercise ST") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# thal 
k2 <- ggplot(df, aes(thal, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Thalium stress test") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# fbp
l2 <- ggplot(df, aes(fbs, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Fasting blood sugar") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# exang
m2 <- ggplot(df, aes(exang, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Exercise induced angina") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

# ca
n2 <- ggplot(df, aes(ca, group = target, fill = target)) +
  geom_bar(position = "dodge") +
  labs(x = "", y = "", title = "Flouroscopy") +
  coord_flip() +
  scale_fill_manual(values = c("green", "red"))+
  theme_minimal() +
  theme(legend.position='none')

ggarrange(g2, h2, i2, j2, k2, l2, m2, n2, 
          ncol = 2, nrow = 4, common.legend = TRUE,
          align = "hv") # It may take a while to print this plot

  
# There are less woman with heart disease in this data set. Women with heart disease have a significantly higher resting blood presure contrary to male with heart disease. Similarly to men, women with heart disease have a lower maximum heart rate in response to the thallium test. 
# 
# For the continuous variables, we can also inspect their mean and standard deviation grouped by gender and disease status as follows:

knitr::kable(df2 %>%
  group_by(target, sex) %>%
  summarise(
    n = n(),
    age = paste(round(mean(age), digits=2)," (",
                round(sd(age), digits=2), ")", sep = ""),
    trestbps = paste(round(mean(trestbps), digits=2), " (",
                     round(sd(trestbps), digits=2), ")", sep = ""),
    chol = paste(round(mean(chol), digits=2), " (",
                 round(sd(chol), digits=2), ")", sep = ""),
    thalach = paste(round(mean(thalach), digits=2), " (",
                    round(sd(thalach), digits=2), ")", sep = ""),
    oldpeak = paste(round(mean(oldpeak), digits=2), " (",
                    round(sd(oldpeak), digits=2), ")", sep = "")
  ), caption = "Mean (SD) of the continous variables grouped by gender and health status.")


# ### 2.3 Correlations  
# To calculate correlations between the variables in the dataset, we'll make use of the function `ggcorr()` applied to the numerical values in the original dataset after removing the NAs.

df <- copy %>%
  filter(thal != 0 & ca != 4) # remove values correspondind to NA in original dataset
    
ggcorr(df, geom = "tile", label = T, label_alpha = T) +
  labs(title = "Correlations between variables in the complete dataset")


# ## 3. Split into train and test datasets  ####
# We will train the models with 80% of samples, leaving out the 20% for test purposes only:  
# Test set will be 20% of original dataset

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = df2$target, times = 1, p = 0.2, list = FALSE)
train <- df[-test_index,]
train$target <- as.factor(train$target)
test <- df[test_index,]
test$target <- as.factor(test$target)

paste("Number of rows in the train dataset: ", nrow(train), sep = "")
paste("Number of rows in the test dataset: ", nrow(test), sep = "")


# ## 4. Building classification models
# In this section we'll compare accuracy and Kappa values for 3 different methods: Random Forest (RF), Decision Trees (DT) and General Linear Model (GLM) with a 5-fold cross-validation within the train dataset. Besides, we'll classify the variables importance for each one of the models.  

### 4.1 Random Forest
fit_rf <- train(target~.,
             data = train,
             method = "rf",
             trControl = trainControl(method = "cv", # Cross-validation
                                      number = 5,    # 5 folds
                                      p = 0.8))

### 4.2 Decision Trees  
fit_rpart <- train(target~.,
             data = train,
             method = "rpart",
             trControl = trainControl(method = "cv", # Cross-validation
                                      number = 5,    # 5 folds
                                      p = 0.8))

### 4.3 General Linear Model  
fit_glm <- train(target~.,
             data = train,
             method = "glmnet",
                 preProc = c("scale","BoxCox"),
             trControl = trainControl(method = "cv", # Cross-validation
                                      number = 5,    # 5 folds
                                      p = 0.8))

# ### 4.4 Evaluating the models  
# 
# Let's take advantage of the `resample()` function to visualize in a boxplot fashion the results yielded by the 3 models trained:  

resamp <- resamples(list(RF = fit_rf,
                        DT = fit_rpart,
                        GLM = fit_glm))
bwplot(resamp, main = "Accuracy and Kappa values for RF, DT and GLM models")

# We can also make use of the `splom()` function to visualize, in a scatter plot matrix, each one of the accuracies yielded during the training process (remember that it was done through a cross-validation with 5 folds)
splom(resamp)

# ## 5. Applying the models ####
# Now let's apply the 3 trained models to the test dataset to try to predict the target variable.

# ### 5.1 Random Forest
# Accuracy of the Random Forest model is as follows:  
pred_rf <- predict(fit_rf, test)
rf_cm <- confusionMatrix(pred_rf, test$target)
rf_cm

# We can see which are the most important features for this model by plotting the results of the `varImp()` function:  
plot(varImp(fit_rf), main = "Variable importance ranking for RF model")

# ### 5.2 Decision Tree
pred_rpart <- predict(fit_rpart, test)
dt_cm <- confusionMatrix(pred_rpart, test$target)
dt_cm

# We can see which are the most important features for this model by plotting the results of the `varImp()` function:  
plot(varImp(fit_rpart), main = "Variable importance ranking for DT model")

# ### 5.3 General Linear Model
pred_glm <- predict(fit_glm, test)
glm_cm <- confusionMatrix(pred_glm, test$target)
glm_cm

# We can see which are the most important features for this model by plotting the results of the `varImp()` function:  
plot(varImp(fit_glm), main = "Variable importance ranking for GLM model")

# The table below summarizes the accuracy and kappa values achieved by each trained model:
sum_table <- as.data.frame(rbind(rf_cm$overall[1:2], 
                   dt_cm$overall[1:2], 
                   glm_cm$overall[1:2]))
row.names(sum_table) <- c("Random Forest", "Decision Tree", "General Linear Model")
kable(sum_table, caption = "Summary of the accuracy and Kappa values achieved.")
# 
#   
# ## 6. Conclusions, limitations and future work
# ### 6.1 Conclusions
# This project aimed at applying the knowledge gained during the Harvard edX Data Science course over a publicly available dataset (Heart Disease UCI dataset, available at Kaggle).  
# 
# After downloading the dataset, we have performed the following operations over it:  
# 
# 1. Transform the dataset into a tidy format, making use of the `tidyverse` library.
# 2. Proceed with a comprehensive data visualization process, differentiating continuous from categorical variables, and grouping by the variables of interest, in this case, gender and health status. For this purpose, we've made an extensive use of several libraries, such as `ggplot2`, `knitr`, `DataExplorer`, `gridExtra`, `ggpubr` and `GGally`.
# 3. Making use of the `caret` library, then we proceeded to generate train and test subsets in order to evaluate the performance of three different machine learning methods when predicting the target variable (health status of a subject).
# 4. These three machine learning methods (Random Forest, Decision Trees and General Linear Model) were then trained and their performance assessed through a 5-fold cross-validation on the train subset making use of the libraries `e1071`, `rpart` and `glmnet` respectively. 
# 5. Finally, we evaluated the trained models against the test subset in terms of accuracy and Kappa values achieved, and visualized the ranking of the variables importance for each one of the models trained.  
# 
# In this project, we've successfully applied the techniques learned during the course and some other libraries and functions have been explored.  
# 
# ### 6.2 Limitations
# This work has two main limitations: on one hand, the dataset is not large enough to extract meaningful conclusions about the correlation between the variables and health status of the subjects. However, it poses a good example about how some clinical variables are more determinant than others when trying to predict the health status of a subject.  
# 
# On the other hand, the dataset is biased, accounting with two times more males than females and a higher population of asymptomatic subjects, which burdens the training capacity of the machine learning models.  
# 
# ### 6.3 Future work
# Future work could address the assessment of other machine learning models, such as Extreme Gradient Boosted Decision Trees (XGBoost), Support Vector Machines (SVM) and/or other clustering methods such as k-Nearest Neighbors (kNN), among others, and compare the accuracy and Kappa values obtained.  
# 
# Besides, it could be further investigated other performance measurements such as the area under the receiver-operator curve (AUC).
# 
# 
