## ACCT337 Statistical Programming
## AY19/20 Term 1
## Group Project

# --------------- # Merlion Credit Risk Models using R # --------------- #

# Install the necessary packages for modelling
install.packages("factoextra")
install.packages("readr")
install.packages("dplyr")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("caret")
install.packages("psych")
install.packages("car")
install.packages("stringr")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("ROSE")
install.packages("pROC")
install.packages("DMwR")
install.packages("mltools")


# Load the necessary Packages
library(factoextra)
library(readr)
library(dplyr)
library(forecast)
library(corrplot)
library(e1071)
library(ggplot2)
library(caret)
library(psych) 
library(car)
library(stringr)
library(rpart)
library(rpart.plot)
library(ROSE)
library(pROC)
library(DMwR)
library(mltools)

# --------------------------------------------------------------------------------- #
# -------------------------- General Data Pre Processing -------------------------- # 
# --------------------------------------------------------------------------------- #

# Formating fund_mt variable as date
merlion.raw <- read_csv("merlion.csv",
                        col_types = cols(fund_mt = col_date(format = "%b-%y")))

# format emp_l as continuous variable 
levels_emp <- c('n/a','< 1 year','1 year','2 years','3 years','4 years',
                '5 years','6 years','7 years','8 years','9 years','>=10 years')

merlion.raw$emp_l <- as.numeric(factor(merlion.raw$emp_l, levels = levels_emp))

# format plan as continuous variable
merlion.raw$plan <- ifelse(merlion.raw$plan == "3 years", 3, 5)

# check whether variables are unique
lengths(lapply(merlion.raw, unique))

# remove unnecessary variables

merlion.sel <- merlion.raw %>%
  select(-c(app_typ, inc_ann_jt, ttl_int_rec, ttl_pr_rec,ttl_pym))

str(merlion.sel)
lengths(lapply(merlion.sel, unique))

# ---------------------------------------------------------------- #
# ------------------ Exploratory Data Analysis ------------------- #
# ---------------------------------------------------------------- #

# Classify "Chargeoff" as Default
merlion.class <- merlion.raw %>% 
  mutate(loan_stat = str_replace_all(merlion.raw$loan_stat, 
                                     c("Chargeoff" = "Default")))

# -- Plot 1: Annual income across credit ratings-- 
# Plot box and whisker of income and ratings 
options(scipen = 999)
ggplot(data = merlion.class, mapping = aes(x = rating, y = inc_ann)) +
  geom_boxplot() + scale_y_log10() + 
  labs(y = "Annual Income", x = "Credit Rating", title = "Annual Income Across Credit Ratings")

# -- Plot 2: Loan amount across time --
# Plot scatter of loan amount and time 
ggplot(merlion.class, aes(x = fund_mt, y = loan_amt)) + geom_point() +
  labs(y = "Loan Amount", x = "Year", title = "Loan Amount Across The Years")

# -- Plot 3: Default to paid ratio --
# Calculate number of default and paid
merlion.value <- count(merlion.class, loan_stat) %>% 
  filter(loan_stat %in% c("Default","Paid")) %>% 
  rename(Loan_Status = loan_stat,
         Percentage = n) 

# Plot stacked bar chart
ggplot(merlion.value, aes(fill = Loan_Status, y = Percentage , x = "Loan Status")) + 
  geom_bar(position = "fill", stat = "identity") + labs(title = "Default to Paid Ratio")

# -- Plot 4: Number of defaults across years -- 
# Show fund_mt as year only in a separate column 
merlion.class.year <- mutate(merlion.class, 
                             year = format(as.Date(merlion.class$fund_mt, format="%d/%m/%Y"),"%Y")) 

# Filter out default data 
merlion.default <- filter(merlion.class.year, loan_stat %in% c("Default")) 

# Count the number of default by year 
merlion.count <- count(merlion.default,year, loan_stat) %>%
  rename(no_of_default = n) 

# Plot line chart
ggplot(data = merlion.count, aes(x = year, y = no_of_default, group = 1)) + 
  geom_line()+ geom_point() + 
  labs(y= "Number of Defaults", x = "Year", title = "No. of Defaults Across the Years")

# ---------------------------------------------------------------- #
# ------------------- Detecting discrepancies -------------------- #
# ---------------------------------------------------------------- #

# Find the difference between payments received
merlion.rec <- merlion.raw %>% 
  mutate(diff_rec = ttl_pym - (ttl_int_rec + ttl_pr_rec))

# Round the column into 2 decimal places
merlion.rec[ ,"diff_rec"] <- round(merlion.rec[ ,"diff_rec"],2) 

# Extracting lines with discrepancies more than $251.51
merlion.flag <- filter(merlion.rec, diff_rec > 251.51)

# Export the file to flag out to management 
write.csv(merlion.flag,'discrepancies.flaggedout.csv')

# ---------------------------------------------------------------- #
# -------------------------- Clustering -------------------------- #
# ---------------------------------------------------------------- #

# Clustering Data Pre-Processing
merlion.drop <- merlion.raw %>%
  select(-c(app_typ, inc_ann_jt, ttl_int_rec, ttl_pr_rec, ttl_pym))

# Removing extreme outlier data
merlion.cluster.sel <- merlion.drop[-c(229413),]

str(merlion.cluster.sel)

# Integer Encoding
merlion.ienc = merlion.cluster.sel %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)

str(merlion.ienc)

# Normalisation of data
merlion.norm = sapply(merlion.ienc,scale)
rownames(merlion.norm) = rownames(merlion.ienc)

# Plotting Elbow Plot to find optimal number of clusters
# Compute and plot wss for k = 2 to k = 15.

set.seed(1)
k.max <- 15
wss <- (nrow(merlion.norm)-1)*sum(apply(merlion.norm,2,var))
for (i in 2:k.max) wss[i] <- sum(kmeans(merlion.norm,
                                        centers=i,
                                        iter.max = 15,
                                        algorithm = "Hartigan-Wong")$withinss)
options(scipen = 999)

# Plotting the Elbow Plot
plot(1:k.max, wss, type="b", xlab="Number of Clusters K",
     ylab="Total within-cluster sum of squares",
     main="Elbow Plot to find Optimal Number of Clusters",
     pch=19, cex=1)

# Based on the Elbow Plot, the optimal K is chosen as 6
# we should label it on the plot using abline()
abline(v = 8, lty = 2)

# Assign the "optimal" number to a variable
Ci = 8

#k means clustering with 8 clusters
km = kmeans(merlion.norm,Ci)

#Cluster Information - number of records in each cluster
km$size #smallest cluster often is the "outlying cluster"

# Cluster Information - Cluster Centroids
km$centers # computer has the corresponding centroid

# Cluster Information - dist between cluster centers
dist(km$centers)


# Assign Cluster Numbers to dataset
merlion.clustn = merlion.ienc %>%
  bind_cols(data.frame(km$cluster))

# Plotting the Cluster
fviz_cluster(km, data=merlion.norm,
             ellipse.type = "convex",
             outlier.color = "black", 
             outlier.shape = 23)


#@$%&@$%&@$%&@$% Outlier detection #@$%&@$%&@$%&@$%

# calculate distances between objects and cluster centers
# this will give us all the corresponding values for every clusters
km.centers = km$centers[km$cluster, ]
km.dist = sqrt(rowSums((merlion.norm - km.centers)^2)) # Euclidean Distance -> distance between the centroid of the same cluster to the every individual points -> use this distance number to calculate and see the outlier

# calculate mean distances by cluster:
mdist.km = tapply(km.dist, km$cluster, mean) # calculate the mean distance for each cluster

# divide each distance by the mean for its cluster:
distscore.km = km.dist/(mdist.km[km$cluster]) # scoring for every single records
distfact.km = data.frame(distscore.km)
colnames(distfact.km) = "DIST.FACTOR"

# Min-Max Normalisation
minmaxscore.km = data.frame((distscore.km - 
                               min(distscore.km))/(max(distscore.km)
                                                   -min(distscore.km)))

colnames(minmaxscore.km) = "SCORE.MINMAX"

# dataframe of dataset with cluster#,distance score,distance score percentile
merlion.clustdn = merlion.clustn %>%
  bind_cols(distfact.km, minmaxscore.km) %>%
  bind_cols(select(merlion.cluster.sel, int_rate, loan_stat, rating))

# Outliers (Top N)

N = 100 #Determine N

# Order datapoints by distance from cluster centers
km.distorder = order(distscore.km, decreasing=TRUE)
merlion.clustdn.order = merlion.clustdn[km.distorder,]
merlion.clustdn.order$rank = rank(-merlion.clustdn.order$SCORE.MINMAX)

# Plot graph to visualise the top 100 outliers
plot(x = merlion.clustdn.order$SCORE.MINMAX,
     y = merlion.clustdn.order$DIST.FACTOR,
     main = "Outliers based on Top N",
     xlab = "MINMAX_SCORE",
     ylab = "Distance Score",
     col = ifelse(merlion.clustdn.order$rank <= N,"orange","black"),
     pch = 14)

km.topN.outl = data.frame(merlion.clustdn.order[merlion.clustdn.order$rank <= N,])

# Export the Top 100 Outlier Customer Loans of the dataset
write.csv(km.topN.outl, file = "Top 100 Outliers.csv",row.names = FALSE)

# Export the Dataset that has their Outlier Ranking and respective clusters
write.csv(merlion.clustdn.order, file = "All.csv",row.names = FALSE)

# ---------------------------------------------------------------- #
# -------------------------- Regression -------------------------- #
# ---------------------------------------------------------------- #

# Part 1: Regression data pre-pocessing
# identify highly skewed variables
summary(merlion.sel) 
skewness(merlion.sel$accts_pastd)
skewness(merlion.sel$bankr_rec)
skewness(merlion.sel$inc_ann)
skewness(merlion.sel$emp_l)
skewness(merlion.sel$mort_ac)
skewness(merlion.sel$loan_amt)

# create function to standardize inn_ann & loan_amt variables
signedlog10 = function(x) {
  ifelse(abs(x) <= 1, 0, sign(x)*log10(abs(x)))
}

# create function to standardize other numerical variables
signedlog2 = function(y) {
  ifelse(abs(y) == 0, 0, sign(y)*log2(abs(y)))
}

# standardize all skewed numeric variables 
merlion.reg <- merlion.sel %>% 
  mutate(log_inc_ann = signedlog10(inc_ann),
         log_accts_pastd = signedlog2(accts_pastd),
         log_bankr_rec= signedlog2(bankr_rec),
         log_emp_l = signedlog2(emp_l),
         log_mort_ac = signedlog2(mort_ac),
         log_loan_amt = signedlog10(loan_amt))

# Part 2: Select variables and records for prediction: ----
# select variables for prediction 
sel.var <- c("log_accts_pastd", "log_bankr_rec","log_emp_l", "log_inc_ann", "log_mort_ac", 
             "log_loan_amt", "purp", "rating", "home_own")

#select records for prediction 
merlion.reg.set <- merlion.reg %>%
  filter(fund_mt >= "2013-01-01") 

# Part 3: Data Partioning: ----
set.seed(1)
index <- sample(c(1:nrow(merlion.reg.set)),
                nrow(merlion.reg.set)*0.7)
trainSet <- merlion.reg.set[index, sel.var]
testSet <- merlion.reg.set[-index, sel.var]

# Part 4: Run lm() function ----
merlion.mlm <- lm(log_loan_amt ~., trainSet)
options(scipen = 999)

summary(merlion.mlm)
plot(merlion.mlm)
# Check for non-linearity properly, if good go further 
# This could only be done after the model was created
crPlots(merlion.mlm) # there is little non-linearity among the variables

# Part 5: Evaluate the model ----
merlion.mlm.pred <- predict(merlion.mlm, testSet)
merlion.mlm.results <- data.frame("Actual" = testSet$log_loan_amt,
                                  "Predicted" = merlion.mlm.pred,
                                  "Errors" = testSet$log_loan_amt - merlion.mlm.pred)

# Evaluate the accurary and correlation of the model 
round(accuracy(merlion.mlm.pred, testSet$log_loan_amt),2)
test.corr <- round(cor(merlion.mlm.pred, testSet$log_loan_amt), 4)

# Check for multi-collinearity using Variance Inflation Factor (VIF) and compare to the correlation matrix 
vif(merlion.mlm)

# correlation matrix 
merlion.sel.ienc <- merlion.sel %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric) %>% 
  select(-fund_mt)

merlion.sel.corr <- cor(merlion.sel.ienc)

corrplot(merlion.sel.corr, method="circle", type = "upper")

#------------------ WITHOUT STANDARDISATION --------------------------

sel.var.nil <- c("accts_pastd", "bankr_rec", "emp_l", "inc_ann", "mort_ac", 
                 "loan_amt", "purp", "rating", "home_own")

# Slit into train & test set: ----
trainSet.nil <- merlion.reg.set[index, sel.var.nil]
testSet.nil <- merlion.reg.set[-index, sel.var.nil]

# Run lm() function ----
merlion.mlm.nil <- lm(loan_amt ~., trainSet.nil)
options(scipen = 999)

summary(merlion.mlm.nil)

# Evaluate the model without standardisation ----
merlion.mlm.pred.nil <- predict(merlion.mlm.nil, testSet.nil)
merlion.mlm.results.nil <- data.frame("Actual" = testSet.nil$loan_amt,
                                      "Predicted" = merlion.mlm.pred.nil,
                                      "Errors" = testSet.nil$loan_amt - merlion.mlm.pred.nil)

# Evaluate the accurary and correlation of the model 
round(accuracy(merlion.mlm.pred.nil, testSet.nil$loan_amt),2)
test.corr <- round(cor(merlion.mlm.pred.nil, testSet.nil$loan_amt), 4)


# -------------------------------------------------------------------- #
# -------------------------- Classification -------------------------- #
# -------------------------------------------------------------------- #

# Loading the Dataset again
merlion.classification.raw = read_csv("merlion.csv")

# Dropping Unnecessary Variables
# Removed 6 vars: app_type, fund_mt, inc_ann_jt, ttl_int_rec, ttl_pr_rec,	ttl_pym
merlion.classification.sel = merlion.classification.raw %>% select(-c(2,5,8,16,17,18))

# Converting Variables to appropriate formats
# Convert "Chr" variable types to "Factor" 
# Bin the emp_l data 
str(merlion.classification.sel)
merlion.bin = merlion.classification.sel %>% 
  mutate(emp_l = case_when(emp_l == 'n/a' ~ "<= 1 year",
                           emp_l == '< 1 year' ~ "<= 1 year", 
                           emp_l == '1 year' ~ "<= 1 year",
                           emp_l == '2 years' ~ "2 - 4 years",
                           emp_l == '3 years' ~ "2 - 4 years",
                           emp_l == '4 years' ~ "2 - 4 years",
                           emp_l == '5 years' ~ "5 - 7 years",
                           emp_l == '6 years' ~ "5 - 7 years",
                           emp_l == '7 years' ~ "5 - 7 years",
                           emp_l == '8 years' ~ "> = 8 years",
                           emp_l == '9 years' ~ "> = 8 years",
                           emp_l == '>=10 years' ~ "> = 8 years")) 
merlion.formatted = merlion.bin %>% mutate_at(c(3,4,8,10,11,12),as.factor)
str(merlion.formatted)

# Splitting Dataset into Training Set, Test Set, and Validation Set
merlion.current = merlion.formatted %>% filter(loan_stat %in% c("Current")) #Validation Set
merlion.non.current = merlion.formatted %>% 
  filter(loan_stat %in% c("Chargeoff", "Default", "Paid")) #Filter for non-current loan status

# Classify loan status of Default and Chargeoff to Default(1) and "Paid" to Non-Default(0)
merlion.non.current.new = mutate(merlion.non.current, 
                                 loan_stat = str_replace_all(merlion.non.current$loan_stat, 
                                                             c("Default" = "1", 
                                                               "Chargeoff" = "1",
                                                               "Paid" = "0"))) 

# Partion Dataset to 60(training)-40(test)
set.seed(1)
train.index = createDataPartition(merlion.non.current.new$loan_stat, p = 0.6, list = FALSE, times = 1)
test.index = createDataPartition(merlion.non.current.new$loan_stat, p = 0.4, list = FALSE, times = 1)
merlion.train = merlion.non.current.new[train.index,] # train set
merlion.test = merlion.non.current.new[test.index,] # test set
table(merlion.train$loan_stat)

# --------------------------- MODELS --------------------------- #

# 1st Model: Undersampling
merlion.undersample <- ovun.sample(loan_stat ~ ., data = merlion.train, method = "under", seed = 1)$data
table(merlion.undersample$loan_stat)
tree.undersample = rpart(loan_stat ~., method = "class",
                         data = merlion.undersample, cp = 0.001, minsplit = 5000)

## Plot Undersample Tree
prp(tree.undersample,
    type = 1,
    extra = 1,
    split.font = 1,
    varlen = -10)

# 2nd Model: Oversampling (Chosen Model)
merlion.oversample <- ovun.sample(loan_stat ~ ., data = merlion.train, method = "over", seed = 1)$data
table(merlion.oversample$loan_stat)
tree.oversample = rpart(loan_stat ~., method = "class",
                        data = merlion.oversample, cp=0.001, minsplit = 5000)

## Plot Over Sampling Tree 
rpart.rules(tree.oversample) #extract rules of classification tree
prp(tree.oversample,
    type = 1,
    extra = 1,
    split.font = 1,
    varlen = -10,
    digits = -1)

# 3rd Model: Loss Matrix
tree.loss.matrix <- rpart(loan_stat ~ ., method = "class",
                          data =  merlion.train, 
                          parms = list(loss = matrix(c(0, 4, 1, 0), ncol = 2)),
                          cp = 0.001, minsplit = 5000)
summary(tree.loss.matrix) 

## Plot Loss Matrix Tree
prp(tree.loss.matrix,
    type = 1,
    extra = 1,
    split.font = 1,
    varlen = -10,
    digits = -1)

# 4th Model: ROSE 
merlion.ROSE = ROSE(loan_stat ~., data = merlion.train, seed = 1)$data
table(merlion.ROSE$loan_stat)
tree.ROSE = rpart(loan_stat ~., method = "class",
                  data = merlion.ROSE, cp=0.001, minsplit = 5000)

## Plot ROSE Tree
prp(tree.ROSE,
    type = 1,
    extra = 1,
    split.font = 1,
    varlen = -10)

# 5th Model: SMOTE
set.seed(1)
merlion.train.SMOTE = merlion.train %>% 
  mutate_if(is.integer, as.factor) %>%
  mutate_if(is.character, as.factor)
merlion.SMOTE = SMOTE(loan_stat ~., as.data.frame(merlion.train.SMOTE))
table(merlion.SMOTE$loan_stat)
tree.SMOTE = rpart(loan_stat ~., method = "class",
                   data = merlion.SMOTE, cp=0.001, minsplit = 5000)

## Plot SMOTE Tree
prp(tree.SMOTE,
    type = 1,
    extra = 1,
    split.font = 1,
    varlen = -10)

# --------------------------- PREDICT --------------------------- #

# Undersampling
undersample.pred = predict(tree.undersample, 
                           merlion.test, 
                           type ="class")

undersample.pred.class = data.frame(undersample.pred)

confusionMatrix(table(undersample.pred, merlion.test$loan_stat),
                positive = "1")

# Oversampling (Chosen Model)
oversample.pred = predict(tree.oversample, 
                          merlion.test, 
                          type ="class")

oversample.pred.class = data.frame(oversample.pred)

confusionMatrix(table(oversample.pred, merlion.test$loan_stat),
                positive = "1")


# Loss Matrix
loss.matrix.pred = predict(tree.loss.matrix, 
                           merlion.test, 
                           type ="class")

loss.matrix.pred.class = data.frame(loss.matrix.pred)

confusionMatrix(table(loss.matrix.pred, merlion.test$loan_stat),
                positive = "1")

# ROSE
ROSE.pred = predict(tree.ROSE, 
                    merlion.test, 
                    type ="class")

ROSE.pred.class = data.frame(ROSE.pred)

confusionMatrix(table(ROSE.pred, merlion.test$loan_stat),
                positive = "1")

# SMOTE
merlion.test.SMOTE = merlion.test %>%
  mutate_if(is.integer, as.factor) %>%
  mutate_if(is.character, as.factor)
SMOTE.pred = predict(tree.SMOTE,
                     merlion.test.SMOTE,
                     type = "class")
SMOTE.pred.class = data.frame(SMOTE.pred)

confusionMatrix(table(SMOTE.pred, merlion.test$loan_stat),
                positive = "1")

# --------------------------- ROC CURVE --------------------------- #
#Oversampling Model (Chosen Model)
oversample.pred.prob = predict(tree.oversample,
                               merlion.test,
                               type="prob")

merlion.combined = cbind(merlion.test,
                         oversample.pred,
                         oversample.pred.prob)

par(mar=c(10,5,1,0.5))

merlion.ROC = plot.roc(merlion.combined$loan_stat,
                       merlion.combined$`1`,
                       main = "ROC for Oversampling Model")

print(merlion.ROC)

mcc(preds =  oversample.pred,
    actuals = as.factor(merlion.combined$loan_stat))

# --------------------------- CURRENT --------------------------- #
# Applying the Predictive Model to the Current Data
merlion.validation = predict(tree.oversample,merlion.current, type = "class")
merlion.validation.class = data.frame(merlion.current,merlion.validation)
merlion.validation.class = rename(merlion.validation.class, "Default(1)/Non-Default(0)" = "merlion.validation")

# Export the predicted results for the Current Data
write.csv(merlion.validation.class,'loan.current.predict.csv')


# --------------------------------------------------------------- #
# ----------------------------- END ----------------------------- # 
# --------------------------------------------------------------- #