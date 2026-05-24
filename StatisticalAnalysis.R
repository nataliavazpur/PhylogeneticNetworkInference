library(ggplot2)
library(dplyr)
library(reshape2)

# List of your binary algorithms
algos <- c("MCTSNNBIN", "MCTSBIN", "TrivialRandBIN", "PureRandBIN")
parametros <- c("leaves", "trees", "Minimum")

# Create a quick correlation matrix
cor_matrix <- round(cor(results[, c(algos, parametros)], method = "spearman"), 3)

# 1. Filter the matrix to see only the Algorithms vs Parameters relationship
# (this way we avoid the correlations of parameters with themselves)
cor_subset <- cor_matrix[algos, parametros]

# 2. Convert the matrix to long format
melted_cor <- melt(cor_subset)

# 3. Create the Heatmap
ggplot(melted_cor, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#e74c3c", high = "#2ecc71", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Spearman\nCorrelation") +
  geom_text(aes(label = value), color = "black", size = 4) + # Add the numbers
  labs(x = "Problem Parameters",
       y = "Algorithms (Binary Success)") +
  theme_minimal()


# Show only the relationship between Algorithms and Parameters
print(cor_matrix[algos, parametros])

# Model for V-MCTS
modelo_vmcts <- glm(MCTSNNBIN ~ leaves + trees + Minimum, 
                    data = results, 
                    family = "binomial")

summary(modelo_vmcts)
# Model for MCTS
modelo_mcts <- glm(MCTSBIN ~ leaves + trees + Minimum, 
                   data = results, 
                   family = "binomial")

summary(modelo_mcts)

# Model for TrivialRand
modelo_trivialrand <- glm(TrivialRandBIN ~ leaves + trees + Minimum, 
                          data = results, 
                          family = "binomial")

summary(modelo_trivialrand)

# Model for PureRand
modelo_purerand <- glm(PureRandBIN ~ leaves + trees + Minimum, 
                       data = results, 
                       family = "binomial")

summary(modelo_purerand)


# Group to get the success rate by zones
heatmap_data <- results %>%
  group_by(leaves, Minimum) %>%
  summarise(Success_Rate = mean(MCTSNNBIN))

ggplot(heatmap_data, aes(x = leaves, y = Minimum, fill = Success_Rate)) +
  geom_tile() +
  scale_fill_gradient(low = "#e74c3c", high = "#2ecc71") +
  labs(title = "                        V-MCTS ",,
       x = "Number of Leaves (n)",
       y = "Upper Bound (k)",
       fill = "Success Rate") +
  theme_minimal()

# Create the contingency table
contingencia <- table(results$MCTSNNBIN, results$MCTSBIN)
colnames(contingencia) <- c("MCTS Fail", "MCTS Success")
rownames(contingencia) <- c("V-MCTS Fail", "V-MCTS Success")

print(contingencia)

# The test
mcnemar.test(contingencia)


# Reshape the data to long format to facilitate comparison across algorithms
df_long <- melt(results, 
                measure.vars = c("MCTSNNBIN", "MCTSBIN", "TrivialRandBIN", "PureRandBIN"),
                variable.name = "Algorithm", 
                value.name = "Success")

# Generate the bar plot for global success rates
ggplot(df_long, aes(x = Algorithm, y = Success, fill = Algorithm)) +
  stat_summary(fun = "mean", geom = "bar") +
  labs(x = "Algorithm",
       y = "Success Rate Proportion") +
  theme_minimal() +
  theme(legend.position = "none") # Optional: remove legend as X-axis already labels bars


algos <- c("MCTSNNBIN", "MCTSBIN", "TrivialRandBIN", "PureRandBIN")
parametros <- c("leaves", "trees", "Minimum")
# 1. Calculate the absolute error for each algorithm (Algorithm Result - Minimum)
# A value of 0 means it found the optimum.
results$error_v_mcts <- results$MCTSNN - results$Minimum
results$error_mcts   <- results$MCTS_old - results$Minimum
results$error_tfrh   <- results$TrivialRand - results$Minimum
results$error_urs    <- results$PureRand - results$Minimum

# 2. Reshape the error data to long format
df_error_long <- melt(results, 
                      measure.vars = c("error_v_mcts", "error_mcts", "error_tfrh", "error_urs"),
                      variable.name = "Algorithm", 
                      value.name = "Error_Distance")

# 3. Generate the Boxplot
ggplot(df_error_long, aes(x = Algorithm, y = Error_Distance, fill = Algorithm)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 1) +
  labs(x = "Algorithm",
       y = "Distance (Retics - Optimal)") +
  theme_minimal() +
  theme(legend.position = "none")


ggplot(df_error_long, aes(x = Algorithm, y = Error_Distance)) +
  # 1. Normal boxplot, but we hide the outliers by default (outlier.shape = NA)
  # We add some transparency (alpha) so the points stand out.
  geom_boxplot(aes(fill = Algorithm), outlier.shape = NA, alpha = 0.5) +
  
  # 2. geom_count counts the repetitions and maps color and size to the count (after_stat(n))
  geom_count(aes(color = after_stat(n))) +
  
  # 3. Color gradient: from blue (few cases) to intense red (many cases)
  scale_color_gradient(low = "blue", high = "red") +
  
  labs(x = "Algorithm",
       y = "Distance (Retics - Optimal)",
       size = "Repetitions",
       color = "Repetitions") +
  theme_minimal() +
  # Keep the count legend, but remove the algorithm fill legend
  guides(fill = "none")

# Generate a table with the exact count of times each error occurs
tabla_errores <- df_error_long %>%
  group_by(Algorithm, Error_Distance) %>%
  summarise(Repeticiones = n(), .groups = "drop") %>%
  # Sort by algorithm and then by the largest errors (outliers)
  arrange(Algorithm, desc(Error_Distance))

# See the results in the console
print(tabla_errores)

ggplot(df_error_long, aes(x = Algorithm, y = Error_Distance)) +
  geom_boxplot(aes(fill = Algorithm), outlier.shape = NA, alpha = 0.5) +
  geom_count(aes(color = after_stat(n))) +
  scale_color_gradient(low = "blue", high = "red") +
  
  # THIS IS THE NEW LINE: Forces the breaks in steps of 1
  scale_y_continuous(breaks = function(x) seq(floor(min(x)), ceiling(max(x)), by = 1)) +
  
  labs(x = "Algorithm",
       y = "Distance (Retics - Optimal)",
       size = "Repetitions",
       color = "Repetitions") +
  theme_minimal() +
  guides(fill = "none")

# TEST COMPARISONS
install.packages("DescTools")
library(DescTools)

# Create a matrix with only the success columns of the 4 algorithms
matriz_exito <- as.matrix(results[, c("MCTSNNBIN", "MCTSBIN", "TrivialRandBIN", "PureRandBIN")])

# Execute Cochran's Q Test
cochran_result <- CochranQTest(matriz_exito)
print(cochran_result)
# Install/load the library for the post-hoc
install.packages("rcompanion")
library(rcompanion)

# Convert the data to long format if you haven't already
library(reshape2)
df_long <- melt(results, 
                measure.vars = c("MCTSNNBIN", "MCTSBIN", "TrivialRandBIN", "PureRandBIN"),
                variable.name = "Algorithm", 
                value.name = "Success")

# Add a row ID so R knows which result belongs to which experiment
df_long$ID <- rep(1:nrow(results), 4)

# Pairwise McNemar Test with Bonferroni correction
post_hoc <- pairwiseMcnemar(Success ~ Algorithm | ID,
                            data    = df_long,
                            method  = "bonferroni")

print(post_hoc)