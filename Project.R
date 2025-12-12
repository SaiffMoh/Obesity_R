rm(list=ls())
options(repos = c(CRAN = "https://cran.rstudio.com/"))

obesity_data <- read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Debug helper: label outputs and add spacing
debug_print <- function(label) {
        cat("\n\n\n[DEBUG] ", label, "\n\n\n", sep="")
}

# Data overview
debug_print("Head of obesity_data")
head(obesity_data)
debug_print("Dimensions of obesity_data")
dim(obesity_data)
debug_print("Column names of obesity_data")
names(obesity_data)
debug_print("Structure of obesity_data")
str(obesity_data)
debug_print("Summary of obesity_data")
summary(obesity_data)

# Check for missing values
debug_print("Column-wise NA counts")
colSums(is.na(obesity_data))

numeric_cols <- c("Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE")

for(col in numeric_cols) {
  if(sum(is.na(obesity_data[[col]])) > 0) {
    obesity_data[[col]][is.na(obesity_data[[col]])] <- mean(obesity_data[[col]], na.rm=TRUE)
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
obesity_data$FCVC <- as.numeric(as.character(obesity_data$FCVC))


obesity_data$BMI <- obesity_data$Weight / (obesity_data$Height^2)

# Group mean comparisons (ANOVA and Tukey HSD)
debug_print("ANOVA: BMI differences across NObeyesdad groups")
anova_bmi <- aov(BMI ~ NObeyesdad, data=obesity_data)
print(summary(anova_bmi))

debug_print("Tukey HSD: Pairwise BMI differences across obesity groups")
print(TukeyHSD(anova_bmi))

debug_print("ANOVA: Age differences across NObeyesdad groups")
anova_age <- aov(Age ~ NObeyesdad, data=obesity_data)
print(summary(anova_age))

debug_print("Tukey HSD: Pairwise Age differences across obesity groups")
print(TukeyHSD(anova_age))

# Exploratory Data Analysis - Visualizations


hist(obesity_data$BMI, 
     col="lightgreen", 
     main="BMI Distribution", 
     xlab="BMI", 
     ylab="Frequency",
     breaks=30)



obesity_counts <- table(obesity_data$NObeyesdad)
par(mar=c(8, 4, 4, 2))
barplot(obesity_counts, 
        col="blue", 
        main="Distribution of Obesity Levels",
        xlab="Obesity Category",
        ylab="Count",
        las=2,
        cex.names=0.7)
par(mar=c(5, 4, 4, 2))

par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(obesity_data$Age, col="lightblue", main="Age", xlab="Age")
hist(obesity_data$Weight, col="lightgreen", main="Weight", xlab="Weight (kg)")
hist(obesity_data$Height, col="coral", main="Height", xlab="Height (m)")
hist(obesity_data$BMI, col="yellow", main="BMI", xlab="BMI")

par(mfrow=c(1,1))

hist(obesity_data$Weight, prob=TRUE, col="grey",
     main="Weight Distribution with Density",
     xlab="Weight (kg)",
     ylim=c(0, max(density(obesity_data$Weight)$y) * 1.1))
lines(density(obesity_data$Weight), col="blue", lwd=2)

hist(obesity_data$Age, prob=TRUE, col="grey",
     main="Age Distribution with Density",
     xlab="Age",
     ylim=c(0, max(density(obesity_data$Age)$y) * 1.1))
lines(density(obesity_data$Age), col="red", lwd=2)


# Correlation Matrix for Numeric Features
numeric_features <- obesity_data[, c("Age", "Height", "Weight", "NCP", "CH2O", "FAF", "TUE", "BMI")]
cor_matrix <- cor(numeric_features)

debug_print("Correlation matrix for numeric features")
print(round(cor_matrix, 3))

# Visualize correlation matrix as heatmap
if(!require(corrplot)) {
  install.packages("corrplot")
}
library(corrplot)

corrplot(cor_matrix, 
         method="color", 
         type="upper",
         addCoef.col="black",
         tl.col="black",
         tl.srt=45,
         number.cex=0.7,
         main="Correlation Matrix - Numeric Features",
         mar=c(0,0,2,0))

plot(obesity_data$Weight, obesity_data$BMI,
     col=as.numeric(obesity_data$NObeyesdad),
     pch=19,
     main="Weight vs BMI by Obesity Level",
     xlab="Weight (kg)",
     ylab="BMI")
legend("topright", legend=levels(obesity_data$NObeyesdad),
       col=1:length(levels(obesity_data$NObeyesdad)),
       pch=19, cex=0.6)

plot(obesity_data$Age, obesity_data$BMI,
     col=as.numeric(obesity_data$NObeyesdad),
     pch=19,
     main="Age vs BMI by Obesity Level",
     xlab="Age",
     ylab="BMI")
legend("topright", legend=levels(obesity_data$NObeyesdad),
       col=1:length(levels(obesity_data$NObeyesdad)),
       pch=19, cex=0.6)


plot(obesity_data$Height, obesity_data$Weight,
     col=as.numeric(obesity_data$NObeyesdad),
     pch=19,
     main="Weight vs Height by Obesity Level",
     xlab="Height (m)",
     ylab="Weight (kg)")
legend("topright", legend=levels(obesity_data$NObeyesdad),
       col=1:length(levels(obesity_data$NObeyesdad)),
       pch=19, cex=0.6)



par(mar=c(9, 4, 4, 2))
boxplot(BMI ~ NObeyesdad, 
        data=obesity_data,
        col=rainbow(7),
        main="BMI Distribution by Obesity Level",
        xlab="Obesity Level",
        ylab="BMI",
        las=2,
        cex.axis=0.75)
par(mar=c(5, 4, 4, 2))

par(mfrow=c(2,3))

gender_counts <- table(obesity_data$Gender)
barplot(gender_counts, 
        col=c("skyblue", "pink"), 
        main="Gender Distribution",
        xlab="Gender",
        ylab="Count")

family_counts <- table(obesity_data$family_history_with_overweight)
barplot(family_counts, 
        col=c("lightgreen", "coral"), 
        main="Family History of Overweight",
        xlab="Family History",
        ylab="Count")

favc_counts <- table(obesity_data$FAVC)
barplot(favc_counts, 
        col=c("yellow", "orange"), 
        main="High Caloric Food Consumption",
        xlab="FAVC",
        ylab="Count")

smoke_counts <- table(obesity_data$SMOKE)
barplot(smoke_counts, 
        col=c("lightgray", "darkgray"), 
        main="Smoking Status",
        xlab="SMOKE",
        ylab="Count")

scc_counts <- table(obesity_data$SCC)
barplot(scc_counts, 
        col=c("lightblue", "darkblue"), 
        main="Calorie Monitoring",
        xlab="SCC",
        ylab="Count")

calc_counts <- table(obesity_data$CALC)
barplot(calc_counts, 
        col=rainbow(length(unique(obesity_data$CALC))), 
        main="Alcohol Consumption",
        xlab="CALC",
        ylab="Count",
        las=2,
        cex.names=0.8)



par(mfrow=c(1,1))

# K-Means Clustering
cluster_data <- obesity_data[, c("Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI")]

set.seed(123)
kc <- kmeans(cluster_data, 7, nstart=25)
debug_print("K-means model summary")
print(kc)

debug_print("Cluster counts from K-means")
print(table(kc$cluster))

cluster_comparison <- table(Cluster=kc$cluster, Obesity=obesity_data$NObeyesdad)
debug_print("Comparison: K-means clusters vs Obesity labels")
print(cluster_comparison)
plot(obesity_data$Height, obesity_data$Weight, 
     col=kc$cluster,
     pch=19,
     main="K-means Clusters (Height vs Weight)",
     xlab="Height (m)",
     ylab="Weight (kg)")
points(kc$centers[,c("Height", "Weight")], col=1:7, pch=8, cex=2)
legend("topright", legend=1:7, col=1:7, pch=19, cex=0.6, title="Cluster")

plot(obesity_data$Age, obesity_data$BMI, 
     col=kc$cluster,
     pch=19,
     main="K-means Clusters (Age vs BMI)",
     xlab="Age",
     ylab="BMI")
points(kc$centers[,c("Age", "BMI")], col=1:7, pch=8, cex=2)
legend("topright", legend=1:7, col=1:7, pch=19, cex=0.6, title="Cluster")

plot(obesity_data$Weight, obesity_data$BMI, 
     col=kc$cluster,
     pch=19,
     main="K-means Clusters (Weight vs BMI)",
     xlab="Weight (kg)",
     ylab="BMI")
points(kc$centers[,c("Weight", "BMI")], col=1:7, pch=8, cex=2)
legend("topright", legend=1:7, col=1:7, pch=19, cex=0.6, title="Cluster")

# Decision Tree Classification
options(repos = c(CRAN = "https://cran.rstudio.com/"))
if(!require(party)) {
  install.packages("party")
}
library(party)

set.seed(123)
ind <- sample(2, nrow(obesity_data), prob=c(0.7, 0.3), replace=TRUE)
train.data <- obesity_data[ind==1,]
test.data <- obesity_data[ind==2,]
obesity.tree <- ctree(NObeyesdad ~ Age + Gender + Height + Weight + 
                      family_history_with_overweight + FAVC + FCVC + 
                      NCP + CAEC + SMOKE + CH2O + SCC + FAF + TUE + 
                      CALC + MTRANS,
                      data=train.data)

# Display tree
debug_print("Decision tree summary (ctree)")
print(obesity.tree)

# Save tree to a file with large dimensions for better visibility
png("decision_tree.png", width=2400, height=1800, res=150)
plot(obesity.tree, main="Decision Tree for Obesity Classification", cex=0.25, type="simple")
dev.off()

cat("\n\n>>> Decision tree saved to 'decision_tree.png' for better viewing <<<\n\n")

tree_pred <- predict(obesity.tree, newdata=test.data)

tree_table <- table(Predicted=tree_pred, Actual=test.data$NObeyesdad)
debug_print("Confusion matrix: Decision Tree")
print(tree_table)

# Visualize confusion matrix as heatmap
par(mfrow=c(1,1))
par(mar=c(10, 10, 4, 2))  # extra margin space for long class labels
image(1:ncol(tree_table), 1:nrow(tree_table), t(as.matrix(tree_table)), 
        col=colorRampPalette(c("white", "lightblue", "darkblue"))(20),
        xlab="Actual", ylab="Predicted", main="Decision Tree Confusion Matrix",
        xaxt="n", yaxt="n")
axis(1, at=1:ncol(tree_table), labels=colnames(tree_table), las=2, cex.axis=0.6)
axis(2, at=1:nrow(tree_table), labels=rownames(tree_table), las=2, cex.axis=0.6)
text(rep(1:ncol(tree_table), each=nrow(tree_table)), 
     rep(1:nrow(tree_table), ncol(tree_table)), 
          as.vector(t(tree_table)), col="black", cex=0.7)
par(mar=c(5, 4, 4, 2))

# Bar plot of correct vs incorrect predictions
correct <- sum(diag(tree_table))
incorrect <- sum(tree_table) - correct
barplot(c(correct, incorrect), 
        names.arg=c("Correct", "Incorrect"),
        col=c("green", "red"),
        main="Decision Tree Predictions",
        ylab="Count",
        ylim=c(0, max(correct, incorrect) * 1.1))
par(mfrow=c(1,1))

tree_accuracy <- sum(diag(tree_table)) / sum(tree_table)

# Support Vector Machine Classification
options(repos = c(CRAN = "https://cran.rstudio.com/"))
if(!require(e1071)) {
  install.packages("e1071")
}
library(e1071)

svm_model <- svm(NObeyesdad ~ Age + Gender + Height + Weight + 
                 family_history_with_overweight + FAVC + FCVC + 
                 NCP + CAEC + SMOKE + CH2O + SCC + FAF + TUE + 
                 CALC + MTRANS,
                 data=train.data, 
                 kernel="radial")

svm_pred <- predict(svm_model, test.data)

svm_table <- table(Predicted=svm_pred, Actual=test.data$NObeyesdad)
debug_print("Confusion matrix: SVM")
print(svm_table)

# Visualize SVM confusion matrix as heatmap
par(mfrow=c(1,1))
par(mar=c(10, 10, 4, 2))
image(1:ncol(svm_table), 1:nrow(svm_table), t(as.matrix(svm_table)), 
      col=colorRampPalette(c("white", "lightgreen", "darkgreen"))(20),
      xlab="Actual", ylab="Predicted", main="SVM Confusion Matrix",
      xaxt="n", yaxt="n")
axis(1, at=1:ncol(svm_table), labels=colnames(svm_table), las=2, cex.axis=0.6)
axis(2, at=1:nrow(svm_table), labels=rownames(svm_table), las=2, cex.axis=0.6)
text(rep(1:ncol(svm_table), each=nrow(svm_table)), 
     rep(1:nrow(svm_table), ncol(svm_table)), 
        as.vector(t(svm_table)), col="black", cex=0.7)
par(mar=c(5, 4, 4, 2))

# Bar plot of correct vs incorrect predictions for SVM
correct_svm <- sum(diag(svm_table))
incorrect_svm <- sum(svm_table) - correct_svm
barplot(c(correct_svm, incorrect_svm), 
        names.arg=c("Correct", "Incorrect"),
        col=c("green", "red"),
        main="SVM Predictions",
        ylab="Count")

svm_accuracy <- sum(diag(svm_table)) / sum(svm_table)

# Naive Bayes Classification
nb_classifier <- naiveBayes(NObeyesdad ~ Age + Gender + Height + Weight + 
                            family_history_with_overweight + FAVC + FCVC + 
                            NCP + CAEC + SMOKE + CH2O + SCC + FAF + TUE + 
                            CALC + MTRANS,
                            data=train.data)

nb_pred <- predict(nb_classifier, test.data)

nb_table <- table(Predicted=nb_pred, Actual=test.data$NObeyesdad)
debug_print("Confusion matrix: Naive Bayes")
print(nb_table)

# Visualize Naive Bayes confusion matrix as heatmap
par(mfrow=c(1,1))
par(mar=c(10, 10, 4, 2))
image(1:ncol(nb_table), 1:nrow(nb_table), t(as.matrix(nb_table)), 
      col=colorRampPalette(c("white", "lightyellow", "orange"))(20),
      xlab="Actual", ylab="Predicted", main="Naive Bayes Confusion Matrix",
      xaxt="n", yaxt="n")
axis(1, at=1:ncol(nb_table), labels=colnames(nb_table), las=2, cex.axis=0.6)
axis(2, at=1:nrow(nb_table), labels=rownames(nb_table), las=2, cex.axis=0.6)
text(rep(1:ncol(nb_table), each=nrow(nb_table)), 
     rep(1:nrow(nb_table), ncol(nb_table)), 
        as.vector(t(nb_table)), col="black", cex=0.7)
par(mar=c(5, 4, 4, 2))

# Bar plot of correct vs incorrect predictions for Naive Bayes
correct_nb <- sum(diag(nb_table))
incorrect_nb <- sum(nb_table) - correct_nb
barplot(c(correct_nb, incorrect_nb), 
        names.arg=c("Correct", "Incorrect"),
        col=c("green", "red"),
        main="Naive Bayes Predictions",
        ylab="Count")

nb_accuracy <- sum(diag(nb_table)) / sum(nb_table)



# Model Comparison
accuracy_comparison <- data.frame(
  Algorithm = c("DT", "SVM", "NB"),
  Accuracy = c(tree_accuracy, svm_accuracy, nb_accuracy)
)

debug_print("Accuracy comparison across classifiers")
print(accuracy_comparison)

# Visualize model comparison
barplot(accuracy_comparison$Accuracy,
        names.arg=accuracy_comparison$Algorithm,
        col=rainbow(3),
        main="Classification Model Accuracy Comparison",
        ylab="Accuracy",
        ylim=c(0,1),
        las=2)


# Linear Regression - BMI Prediction
lm_model <- lm(BMI ~ Age + Weight + Height + FCVC + NCP + CH2O + FAF + TUE,
               data=train.data)

debug_print("Linear model summary: BMI prediction")
print(summary(lm_model))

lm_pred <- predict(lm_model, newdata=test.data)

correlation <- cor(lm_pred, test.data$BMI)

# Actual vs Predicted BMI plot
plot(test.data$BMI, lm_pred,
     main="Actual vs Predicted BMI",
     xlab="Actual BMI",
     ylab="Predicted BMI",
     pch=19,
     col="blue")
abline(0, 1, col="red", lwd=2)


# Additional Analysis - Correlations
cor_age_faf <- cor(obesity_data$Age, obesity_data$FAF)
debug_print("Correlation: Age vs FAF")
cat("Correlation:", round(cor_age_faf, 4), "\n\n\n")

cor_water_bmi <- cor(obesity_data$CH2O, obesity_data$BMI)
debug_print("Correlation: CH2O (water) vs BMI")
cat("Correlation:", round(cor_water_bmi, 4), "\n\n\n")

avg_bmi_transport <- aggregate(obesity_data$BMI, 
                               list(obesity_data$MTRANS), 
                               mean)
names(avg_bmi_transport) <- c("Transportation", "Avg_BMI")
debug_print("Average BMI per transportation method")
print(avg_bmi_transport)

# Barplot of average BMI by transportation
barplot(avg_bmi_transport$Avg_BMI,
        names.arg=avg_bmi_transport$Transportation,
        col=rainbow(nrow(avg_bmi_transport)),
        main="Average BMI by Transportation Method",
        ylab="Average BMI",
        las=2)

write.csv(obesity_data, "obesity_cleaned.csv", row.names=FALSE)