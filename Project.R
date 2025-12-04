# ============================================
# Big Data Analytics Project
# Obesity Levels Estimation Based on Eating Habits
# ============================================

# Clear workspace
rm(list=ls())

# Set CRAN mirror for package installation
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# Set working directory (adjust as needed)
# setwd("your/project/path")

# ============================================
# 1. DATA LOADING AND EXPLORATION
# ============================================

# Load the dataset
obesity_data <- read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# View first few rows
head(obesity_data)

# Check dataset dimensions
cat("Dataset Dimensions:\n")
dim(obesity_data)

# Check column names
cat("\nColumn Names:\n")
names(obesity_data)

# Check data structure
cat("\nData Structure:\n")
str(obesity_data)

# Summary statistics
cat("\nSummary Statistics:\n")
summary(obesity_data)

# ============================================
# 2. DATA CLEANING AND PREPROCESSING
# ============================================

# Check for missing values
cat("\nMissing Values:\n")
colSums(is.na(obesity_data))

# Replace any NA values if they exist
# For numeric columns, replace with mean
numeric_cols <- c("Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE")

for(col in numeric_cols) {
  if(sum(is.na(obesity_data[[col]])) > 0) {
    obesity_data[[col]][is.na(obesity_data[[col]])] <- mean(obesity_data[[col]], na.rm=TRUE)
  }
}

# Check for negative values and fix them
for(col in numeric_cols) {
  if(any(obesity_data[[col]] < 0, na.rm=TRUE)) {
    obesity_data[[col]][obesity_data[[col]] < 0] <- obesity_data[[col]][obesity_data[[col]] < 0] * (-1)
  }
}

# Convert categorical variables to factors
obesity_data$Gender <- as.factor(obesity_data$Gender)
obesity_data$family_history_with_overweight <- as.factor(obesity_data$family_history_with_overweight)
obesity_data$FAVC <- as.factor(obesity_data$FAVC)
obesity_data$CAEC <- as.factor(obesity_data$CAEC)
obesity_data$SMOKE <- as.factor(obesity_data$SMOKE)
obesity_data$SCC <- as.factor(obesity_data$SCC)
obesity_data$CALC <- as.factor(obesity_data$CALC)
obesity_data$MTRANS <- as.factor(obesity_data$MTRANS)
obesity_data$NObeyesdad <- as.factor(obesity_data$NObeyesdad)

# Create BMI variable
obesity_data$BMI <- obesity_data$Weight / (obesity_data$Height^2)

cat("\nData cleaning completed!\n")

# ============================================
# 3. HYPOTHESIS TESTING
# ============================================

cat("\n============================================\n")
cat("HYPOTHESIS TESTING\n")
cat("============================================\n")

# Hypothesis 1: Does gender affect obesity levels?
cat("\nHypothesis 1: Gender vs Obesity Levels\n")
gender_obesity_table <- table(obesity_data$Gender, obesity_data$NObeyesdad)
print(gender_obesity_table)
chisq_test1 <- chisq.test(gender_obesity_table)
print(chisq_test1)

# Hypothesis 2: Does family history affect obesity?
cat("\nHypothesis 2: Family History vs Obesity Levels\n")
family_obesity_table <- table(obesity_data$family_history_with_overweight, obesity_data$NObeyesdad)
print(family_obesity_table)
chisq_test2 <- chisq.test(family_obesity_table)
print(chisq_test2)

# Hypothesis 3: Does high caloric food consumption affect obesity?
cat("\nHypothesis 3: High Caloric Food Consumption vs Obesity\n")
favc_obesity_table <- table(obesity_data$FAVC, obesity_data$NObeyesdad)
print(favc_obesity_table)
chisq_test3 <- chisq.test(favc_obesity_table)
print(chisq_test3)

# ============================================
# 4. EXPLORATORY DATA ANALYSIS (VISUALIZATIONS)
# ============================================

cat("\n============================================\n")
cat("EXPLORATORY DATA ANALYSIS\n")
cat("============================================\n")

# Set up for multiple plots
par(mfrow=c(1,1))

# Visualization 1: Histogram of Age Distribution
hist(obesity_data$Age, 
     col="lightblue", 
     main="Age Distribution", 
     xlab="Age", 
     ylab="Frequency",
     breaks=20)

# Visualization 2: Histogram of BMI Distribution
hist(obesity_data$BMI, 
     col="lightgreen", 
     main="BMI Distribution", 
     xlab="BMI", 
     ylab="Frequency",
     breaks=30)

# Visualization 3: Boxplot for Weight
boxplot(obesity_data$Weight, 
        col=rainbow(1), 
        main="Weight Distribution",
        ylab="Weight (kg)")
rug(obesity_data$Weight, side=2)

# Visualization 4: Boxplot for Height
boxplot(obesity_data$Height, 
        col=rainbow(1), 
        main="Height Distribution",
        ylab="Height (m)")
rug(obesity_data$Height, side=2)

# Visualization 5: Bar plot of Obesity Levels
obesity_counts <- table(obesity_data$NObeyesdad)
barplot(obesity_counts, 
        col=rainbow(length(obesity_counts)), 
        main="Distribution of Obesity Levels",
        xlab="Obesity Category",
        ylab="Count",
        las=2)

# Visualization 6: Bar plot of Gender Distribution
gender_counts <- table(obesity_data$Gender)
barplot(gender_counts, 
        col=c("pink", "lightblue"), 
        main="Gender Distribution",
        xlab="Gender",
        ylab="Count")

# Visualization 7: Physical Activity Frequency
faf_counts <- table(obesity_data$FAF)
barplot(faf_counts, 
        col="orange", 
        main="Physical Activity Frequency",
        xlab="Frequency (0-3)",
        ylab="Count")

# Visualization 8: Transportation Method
transport_counts <- table(obesity_data$MTRANS)
barplot(transport_counts, 
        col=rainbow(length(transport_counts)), 
        main="Transportation Method Distribution",
        xlab="Transportation Type",
        ylab="Count",
        las=2)

# Visualization 9: Density Plot for Weight
hist(obesity_data$Weight, prob=TRUE, col="grey",
     main="Weight Distribution with Density",
     xlab="Weight (kg)")
lines(density(obesity_data$Weight), col="blue", lwd=2)

# Visualization 10: Density Plot for Age
hist(obesity_data$Age, prob=TRUE, col="grey",
     main="Age Distribution with Density",
     xlab="Age")
lines(density(obesity_data$Age), col="red", lwd=2)

# Visualization 11: Scatter plot - Weight vs Height
plot(obesity_data$Height, obesity_data$Weight,
     col=as.numeric(obesity_data$NObeyesdad),
     pch=19,
     main="Weight vs Height by Obesity Level",
     xlab="Height (m)",
     ylab="Weight (kg)")

# Visualization 12: Scatter plot - Age vs BMI
plot(obesity_data$Age, obesity_data$BMI,
     col=as.numeric(obesity_data$NObeyesdad),
     pch=19,
     main="Age vs BMI by Obesity Level",
     xlab="Age",
     ylab="BMI")

# Visualization 13: Boxplot - BMI by Obesity Level
boxplot(BMI ~ NObeyesdad, 
        data=obesity_data,
        col=rainbow(7),
        main="BMI Distribution by Obesity Level",
        xlab="Obesity Level",
        ylab="BMI",
        las=2)

# Visualization 14: Family History Distribution
family_counts <- table(obesity_data$family_history_with_overweight)
barplot(family_counts, 
        col=c("lightgreen", "coral"), 
        main="Family History of Overweight",
        xlab="Family History",
        ylab="Count")

# Visualization 15: Multiple plots together
par(mfrow=c(2,2))
hist(obesity_data$Age, col="lightblue", main="Age", xlab="Age")
hist(obesity_data$Weight, col="lightgreen", main="Weight", xlab="Weight (kg)")
hist(obesity_data$Height, col="coral", main="Height", xlab="Height (m)")
hist(obesity_data$BMI, col="yellow", main="BMI", xlab="BMI")
par(mfrow=c(1,1))

# ============================================
# 5. K-MEANS CLUSTERING
# ============================================

cat("\n============================================\n")
cat("K-MEANS CLUSTERING ANALYSIS\n")
cat("============================================\n")

# Prepare data for clustering (numeric features only)
cluster_data <- obesity_data[, c("Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI")]

# Apply k-means with k=7 (number of obesity categories)
set.seed(123)
kc <- kmeans(cluster_data, 7, nstart=25)

# Display clustering results
cat("\nK-means clustering results:\n")
print(kc)

# Cluster assignments
cat("\nCluster sizes:\n")
print(table(kc$cluster))

# Compare clusters with actual obesity levels
cat("\nComparison of Clusters with Obesity Levels:\n")
cluster_comparison <- table(Cluster=kc$cluster, Obesity=obesity_data$NObeyesdad)
print(cluster_comparison)

# Visualize clusters using Weight and Height
plot(obesity_data$Height, obesity_data$Weight, 
     col=kc$cluster,
     pch=19,
     main="K-means Clusters (Height vs Weight)",
     xlab="Height (m)",
     ylab="Weight (kg)")
points(kc$centers[,c("Height", "Weight")], col=1:7, pch=8, cex=2)

# Visualize clusters using Age and BMI
plot(obesity_data$Age, obesity_data$BMI, 
     col=kc$cluster,
     pch=19,
     main="K-means Clusters (Age vs BMI)",
     xlab="Age",
     ylab="BMI")
points(kc$centers[,c("Age", "BMI")], col=1:7, pch=8, cex=2)

# ============================================
# 6. CLASSIFICATION - DECISION TREE
# ============================================

cat("\n============================================\n")
cat("DECISION TREE CLASSIFICATION\n")
cat("============================================\n")

# Set CRAN mirror and install required package
options(repos = c(CRAN = "https://cran.rstudio.com/"))
if(!require(party)) {
  install.packages("party")
}
library(party)

# Split data into training and testing sets (70-30 split)
set.seed(123)
ind <- sample(2, nrow(obesity_data), prob=c(0.7, 0.3), replace=TRUE)
train.data <- obesity_data[ind==1,]
test.data <- obesity_data[ind==2,]

cat("\nTraining set size:", nrow(train.data))
cat("\nTest set size:", nrow(test.data))

# Build decision tree model
obesity.tree <- ctree(NObeyesdad ~ Age + Gender + Height + Weight + 
                      family_history_with_overweight + FAVC + FCVC + 
                      NCP + CAEC + SMOKE + CH2O + SCC + FAF + TUE + 
                      CALC + MTRANS,
                      data=train.data)

# Display tree
print(obesity.tree)
plot(obesity.tree, main="Decision Tree for Obesity Classification")

# Make predictions
tree_pred <- predict(obesity.tree, newdata=test.data)

# Confusion matrix
cat("\n\nDecision Tree Confusion Matrix:\n")
tree_table <- table(Predicted=tree_pred, Actual=test.data$NObeyesdad)
print(tree_table)

# Calculate accuracy
tree_accuracy <- sum(diag(tree_table)) / sum(tree_table)
cat("\nDecision Tree Accuracy:", round(tree_accuracy, 4))

# ============================================
# 7. CLASSIFICATION - SVM
# ============================================

cat("\n\n============================================\n")
cat("SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION\n")
cat("============================================\n")

# Set CRAN mirror and install required package
options(repos = c(CRAN = "https://cran.rstudio.com/"))
if(!require(e1071)) {
  install.packages("e1071")
}
library(e1071)

# Build SVM model
svm_model <- svm(NObeyesdad ~ Age + Gender + Height + Weight + 
                 family_history_with_overweight + FAVC + FCVC + 
                 NCP + CAEC + SMOKE + CH2O + SCC + FAF + TUE + 
                 CALC + MTRANS,
                 data=train.data, 
                 kernel="radial")

# Make predictions
svm_pred <- predict(svm_model, test.data)

# Confusion matrix
cat("\nSVM Confusion Matrix:\n")
svm_table <- table(Predicted=svm_pred, Actual=test.data$NObeyesdad)
print(svm_table)

# Calculate accuracy
svm_accuracy <- sum(diag(svm_table)) / sum(svm_table)
cat("\nSVM Accuracy:", round(svm_accuracy, 4))

# ============================================
# 8. CLASSIFICATION - NAIVE BAYES
# ============================================

cat("\n\n============================================\n")
cat("NAIVE BAYES CLASSIFICATION\n")
cat("============================================\n")

# Build Naive Bayes model (e1071 already loaded)
nb_classifier <- naiveBayes(NObeyesdad ~ Age + Gender + Height + Weight + 
                            family_history_with_overweight + FAVC + FCVC + 
                            NCP + CAEC + SMOKE + CH2O + SCC + FAF + TUE + 
                            CALC + MTRANS,
                            data=train.data)

# Make predictions
nb_pred <- predict(nb_classifier, test.data)

# Confusion matrix
cat("\nNaive Bayes Confusion Matrix:\n")
nb_table <- table(Predicted=nb_pred, Actual=test.data$NObeyesdad)
print(nb_table)

# Calculate accuracy
nb_accuracy <- sum(diag(nb_table)) / sum(nb_table)
cat("\nNaive Bayes Accuracy:", round(nb_accuracy, 4))

# ============================================
# 9. LINEAR REGRESSION - BMI PREDICTION
# ============================================

cat("\n\n============================================\n")
cat("LINEAR REGRESSION - BMI PREDICTION\n")
cat("============================================\n")

# Build linear regression model to predict BMI
lm_model <- lm(BMI ~ Age + Weight + Height + FCVC + NCP + CH2O + FAF + TUE,
               data=train.data)

# Display model summary
cat("\nLinear Regression Model Summary:\n")
print(summary(lm_model))

# Make predictions
lm_pred <- predict(lm_model, newdata=test.data)

# Calculate correlation between predicted and actual
correlation <- cor(lm_pred, test.data$BMI)
cat("\nCorrelation between predicted and actual BMI:", round(correlation, 4))

# Plot actual vs predicted
plot(test.data$BMI, lm_pred,
     main="Actual vs Predicted BMI",
     xlab="Actual BMI",
     ylab="Predicted BMI",
     pch=19,
     col="blue")
abline(0, 1, col="red", lwd=2)

# ============================================
# 10. MODEL COMPARISON
# ============================================

cat("\n\n============================================\n")
cat("MODEL PERFORMANCE COMPARISON\n")
cat("============================================\n")

# Create comparison table
accuracy_comparison <- data.frame(
  Algorithm = c("Decision Tree", "SVM", "Naive Bayes"),
  Accuracy = c(tree_accuracy, svm_accuracy, nb_accuracy)
)

print(accuracy_comparison)

# Visualize model comparison
barplot(accuracy_comparison$Accuracy,
        names.arg=accuracy_comparison$Algorithm,
        col=rainbow(3),
        main="Classification Model Accuracy Comparison",
        ylab="Accuracy",
        ylim=c(0,1),
        las=2)

# ============================================
# 11. ADDITIONAL ANALYSIS
# ============================================

# Correlation between Age and Physical Activity
cat("\n\nCorrelation between Age and Physical Activity Frequency:\n")
cor_age_faf <- cor(obesity_data$Age, obesity_data$FAF)
cat("Correlation:", round(cor_age_faf, 4))

# Correlation between Water Consumption and BMI
cat("\n\nCorrelation between Water Consumption and BMI:\n")
cor_water_bmi <- cor(obesity_data$CH2O, obesity_data$BMI)
cat("Correlation:", round(cor_water_bmi, 4))

# Average BMI by Transportation Method
cat("\n\nAverage BMI by Transportation Method:\n")
avg_bmi_transport <- aggregate(obesity_data$BMI, 
                               list(obesity_data$MTRANS), 
                               mean)
names(avg_bmi_transport) <- c("Transportation", "Avg_BMI")
print(avg_bmi_transport)

# Barplot of average BMI by transportation
barplot(avg_bmi_transport$Avg_BMI,
        names.arg=avg_bmi_transport$Transportation,
        col=rainbow(nrow(avg_bmi_transport)),
        main="Average BMI by Transportation Method",
        ylab="Average BMI",
        las=2)

cat("\n\n============================================\n")
cat("ANALYSIS COMPLETE!\n")
cat("============================================\n")

# Save cleaned data
write.csv(obesity_data, "obesity_cleaned.csv", row.names=FALSE)
cat("\nCleaned data saved to 'obesity_cleaned.csv'\n")