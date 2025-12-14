library(lme4)
library(lmerTest)
library(dplyr)
library(emmeans)
library(tidyverse)
library(ggplot2)
library(ggsignif)
library(tidytext)
library(RColorBrewer)
library(stringr)
library(grid)
library(caret)
library(mgcv)
library(brms)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
## frequency dataset ----
frequency.data <- read.csv("../../../data/eliciture_regions_surprisal.csv", header=TRUE) %>% 
  select(word, region_surprisal_nats) %>% 
  rename("region_surprisal"= region_surprisal_nats)

## GPT-2 ----
rc_rt.gpt2.data <- read.csv("../../../data/rc_rt/rc_rt_combined_gpt2_3.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.gpt2.data <- rc_rt.gpt2.data %>% 
  left_join(frequency.data, by="word") %>%
  select(-c("frequency", "log_freq", "critical_token"))

# perplexity based on the stimuli in this exp
gpt2_perplexity <- 5.536873862777304
# perplexity based on natural stories (futrell et al. 2021) for sanity check
# gpt2_perplexity <- 4.359453969447

rc_rt.gpt2.summary <- rc_rt.gpt2.data %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                  "RC_VERB+4", "RC_VERB+5")))

## Llama3.2-1B ----
rc_rt.llama1B.data <- read.csv("../../../data/rc_rt/rc_rt_combined_Llama-3.2-1B_1.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.llama1B.data <- rc_rt.llama1B.data %>% 
  left_join(frequency.data, by=c("word")) %>% 
  select(-c("frequency", "log_freq"))

# perplexity based on the stimuli in this exp
llama1B_perplexity <- 4.997065356674766
# perplexity based on natural stories (futrell et al. 2021) for sanity check
# llama1B_perplexity <- 3.7358035353811445

rc_rt.llama1B.summary <- rc_rt.llama1B.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit = fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                    "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                    "RC_VERB+4", "RC_VERB+5")))

## Llama3.2-3B ----
rc_rt.llama3B.data <- read.csv("../../../data/rc_rt/rc_rt_combined_Llama-3.2-3B_1.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.llama3B.data <- rc_rt.llama3B.data %>% 
  left_join(frequency.data, by=c("word")) %>% 
  select(-c("frequency", "log_freq"))

# perplexity based on the stimuli in this exp
llama3B_perplexity <- 4.922765130005042
# perplexity based on natural stories (futrell et al. 2021) for sanity check
# llama3B_perplexity <- 3.599889468050133

rc_rt.llama3B.summary <- rc_rt.llama3B.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit = fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                    "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                    "RC_VERB+4", "RC_VERB+5")))

## Llama3.2-1B-Instruct ----
rc_rt.llama1B_instruct.data <- read.csv("../../../data/rc_rt/rc_rt_combined_Llama-3.2-1B-Instruct_1.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.llama1B_instruct.data <- rc_rt.llama1B_instruct.data %>% 
  left_join(frequency.data, by=c("word")) %>% 
  select(-c("frequency", "log_freq"))

# perplexity based on the stimuli in this exp
llama1B_instruct_perplexity <- 5.011721920477061
# perplexity based on natural stories (futrell et al. 2021) for sanity check
# llama1B_instruct_perplexity <- 3.992406736140558

rc_rt.llama1B_instruct.summary <- rc_rt.llama1B_instruct.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit = fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                    "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                    "RC_VERB+4", "RC_VERB+5")))

## Llama3.2-3B-Instruct ----
rc_rt.llama3B_instruct.data <- read.csv("../../../data/rc_rt/rc_rt_combined_Llama-3.2-3B-Instruct_1.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.llama3B_instruct.data <- rc_rt.llama3B_instruct.data %>% 
  left_join(frequency.data, by=c("word")) %>% 
  select(-c("frequency", "log_freq"))

# perplexity based on stimuli in this exp
llama3B_instruct_perplexity <- 4.868213636262412
# perplexity based on natural stories (futrell et al. 2021) for sanity check
# llama3B_instruct_perplexity <- 3.824219356984297

rc_rt.llama3B_instruct.summary <- rc_rt.llama3B_instruct.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit = fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO", 
                                    "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                    "RC_VERB+4", "RC_VERB+5")))

# 3. Analysis ----
get_predictions <- function(model, data_x) {
  predictions <- predict(mode, data_x,type="response", se.fit=TRUE)
  return(predictions)
}

get_loglikelihood <- function(model, df_x, data_y) {
  " df_x: the dataframe that contains the values to predict from (i.e., the dataframe with surprisal)
    data_y: the ground truth of the predicted value (i.e., the column of reading time)
  "
  predictions <- predict(model, df_x, type="response", se.fit=TRUE)
  residuals <- df_x$surprisal - predictions[["fit"]]
  sigma2 <- mean(residuals^2)
  loglikelihood <- sum(dnorm(data_y, predictions[["fit"]], sqrt(sigma2), log=TRUE))
  return(loglikelihood)
} 

get_prev <- function(dataframe, num_prev) {
  prev_df <- dataframe %>% 
    group_by(item_id, cond) %>% 
    mutate(prev1_surp = lag(surprisal, order_by = crit),
           prev1_word = lag(word, order_by=crit),
           prev1_wordlen = lag(wordlen, order_by=crit),
           prev1_region_surprisal = lag(region_surprisal, order_by = crit),
           
           prev2_surp = lag(prev1_surp, order_by = crit),
           prev2_word = lag(prev1_word, order_by = crit),
           prev2_wordlen = lag(prev1_wordlen, order_by = crit),
           prev2_region_surprisal = lag(region_surprisal, order_by = crit))
  if(num_prev==3) { # if getting the measures of the previous three words 
    prev_df <- prev_df %>% 
      mutate(prev3_surp = lag(prev2_surp, order_by = crit),
             prev3_word = lag(prev2_word, order_by = crit),
             prev3_wordlen = lag(prev2_wordlen, order_by = crit),
             prev3_region_surprisal = lag(region_surprisal, order_by = crit))  %>%
      select(-c(prev_word,prev2_word,prev3_word)) %>%
      drop_na() %>% 
      ungroup()
  } else {
    prev_df <- prev_df %>% 
      select(-c(prev1_word,prev2_word)) %>%
      drop_na() %>% 
      ungroup()
  }
  return(prev_df)
}

get_new_df <- function(dataframe, length_out, num_prev) {
  new_df <- data.frame(surprisal = seq(min(dataframe$surprisal),
                                       max(dataframe$surprisal),
                                       length.out=length_out),
                       prev1_surp = seq(min(dataframe$prev1_surp),
                                        max(dataframe$prev1_surp),
                                        length.out=length_out),
                       prev2_surp = seq(min(dataframe$prev2_surp),
                                        max(dataframe$prev2_surp),
                                        length.out = length_out),
                       wordlen = seq(min(dataframe$wordlen), 
                                     max(dataframe$wordlen), 
                                     length.out = length_out), 
                       prev1_wordlen = seq(min(dataframe$prev1_wordlen),
                                           max(dataframe$prev1_wordlen),
                                           length.out = length_out), 
                       prev2_wordlen = seq(min(dataframe$prev2_wordlen),
                                           max(dataframe$prev2_wordlen),
                                           length.out = length_out),
                       region_surprisal = seq(min(dataframe$region_surprisal),
                                      max(dataframe$region_surprisal),
                                      length.out = length_out),
                       prev1_region_surprisal = seq(min(dataframe$prev1_region_surprisal),
                                                    max(dataframe$prev1_region_surprisal),
                                                    length.out = length_out),
                       prev2_region_surprisal = seq(min(dataframe$region_surprisal),
                                                    max(dataframe$region_surprisal),
                                                    length.out = length_out))
  if(num_prev==3) {
    new_df <- new_df %>% 
      mutate(prev3_surp = seq(min(dataframe$prev3_surp),
                              max(dataframe$prev3_surp),
                              length.out = length_out), 
             prev3_wordlen = seq(min(dataframe$prev3_wordlen),
                                 max(dataframe$prev3_wordlen),
                                 length.out=length_out),
             prev3_region_surprisal = seq(min(dataframe$prev3_region_surprisal),
                                          max(dataframe$prev3_region_surprisal),
                                          length.out = length_out))
  }
  return(new_df)
}

plot_predictions <- function(new_df, predictions_df) {
  ggplot()+
    geom_line(data=data.frame(surprisal=new_df$surprisal,
                              mean_rt=predictions_df$fit),
              aes(x=surprisal,
                  y=mean_rt), size=1)+
    geom_ribbon(data=data.frame(surprisal=new_df$surprisal,
                                fit=predictions_df$fit,
                                se=predictions_df$se.fit),
                aes(x=surprisal,
                    ymin=fit-1.96*se,
                    ymax=fit+1.96*se),
                alpha=0.3)+ 
    labs(x="Surprisal",
         y="Reading time")
}

## GAM models ----
# using the previous three words
baseline_gam = mean_rt ~ te(wordlen, bs="cr") +
  te(prev1_wordlen, bs = "cr") +
  te(prev2_wordlen, bs = "cr") + 
  te(region_surprisal, bs = "cr") + # this is unigram surprisal
  te(prev1_region_surprisal, bs = "cr") +
  te(prev2_region_surprisal, bs = "cr")

full_gam = mean_rt ~ s(surprisal, bs="cr", k=5) + 
  s(prev1_surp, bs="cr", k=5) + 
  s(prev2_surp, bs="cr", k=5) +
  te(wordlen, bs="cr") + 
  te(prev1_wordlen, bs="cr") + 
  te(prev2_wordlen, bs="cr") + 
  te(region_surprisal, bs="cr") +
  te(prev1_region_surprisal, bs = "cr") +
  te(prev2_region_surprisal, bs = "cr")

loglik_testset <- function(y_obs, y_pred, sigma) {
  sum(dnorm(y_obs, mean = y_pred, sd = sigma, log = TRUE))
}

# 10-fold cross validation
cv_gam <- function(full_formula, simple_formula, data, k=10) {
  set.seed(1024)
  n <- nrow(data)
  folds <- sample(rep(1:k, length.out = n))
  
  # shared sigma estimated once
  base_fit <- gam(simple_formula, data=data)
  sigma_shared <- sqrt(mean(residuals(base_fit)^2))
  
  rmse_vec <- numeric(k)
  delta_ll <- numeric(k)
  
  for (i in 1:k) {
    train <- data[folds != i, ]
    test <- data[folds == i, ]
    full_model <- gam(full_formula, data=train)
    full_preds <- predict(full_model, newdata=test, type="response")
    full_sigma <- sqrt(mean(residuals(full_model)^2))
    full_loglike <- loglik_testset(test$mean_rt, full_preds, sigma_shared)
    
    simple_model <- gam(simple_formula, data=train)
    simple_preds <- predict(simple_model, newdata=test, type="response")
    simple_sigma <- sqrt(mean(residuals(simple_model)^2))
    simple_loglike <- loglik_testset(test$mean_rt, simple_preds, sigma_shared)
    
    delta_ll[i] <- (full_loglike - simple_loglike) / nrow(test)
    rmse_vec[i] <- sqrt(mean((test$mean_rt - full_preds)^2))
  }
  return(list(delta_ll=delta_ll, rmse_vec = rmse_vec))
}

### GPT2 ----
rc_rt.gpt2.data <- rc_rt.gpt2.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO", 
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                  "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.gpt2.data <- get_prev(rc_rt.gpt2.data,2)

# gam model
gpt2_model <- gam(full_gam, data=rc_rt_prev.gpt2.data)
summary(gpt2_model)
gam.check(gpt2_model)

# base gam model
gam_base_model <- gam(baseline_gam, data=rc_rt_prev.gpt2.data)
summary(gam_base_model)

## model comparison 
gpt2_gam_ll <- logLik.gam(gpt2_model)
gam_base_ll <- logLik.gam(gam_base_model)
gpt2_gam_delta_ll <- gpt2_gam_ll - gam_base_ll
gpt2_gam_delta_ll # 11.609

## cross-validation
gpt2_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.gpt2.data ,k=10)
rmse_gpt2_gam_cv <- gpt2_gam_cv$rmse_vec
delta_ll_gpt2_gam_cv <- gpt2_gam_cv$delta_ll
delta_ll_gpt2_gam_cv

## visualization
# create the dataframe for prediction and ploting
gpt2_prob_new_data <- get_new_df(rc_rt_prev.gpt2.data,100,2)
# predict using the new dataframe
gpt2_predictions <- predict(gpt2_model, newdata=gpt2_prob_new_data,type="response",
                            se.fit=TRUE)
# plot the model
gpt2_rt_graph <- plot_predictions(gpt2_prob_new_data, gpt2_predictions)
gpt2_rt_graph
ggsave(gpt2_rt_graph, file="../graphs/gpt2_rt_graph.pdf", width=8, height=4)

# plot predictions against actual
gpt2_predictions <- predict(gpt2_model, newdata=rc_rt_prev.gpt2.data,type="response",
                            se.fit=TRUE)
ggplot(data=data.frame(actual_rt=rc_rt_prev.gpt2.data$mean_rt,
                       predicted_rt=gpt2_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="Actual reading time",
       y="Predicted reading time")

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.gpt2.data <- rc_rt.gpt2.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.gpt2.data <- get_prev(rc_rt.gpt2.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit the gam model on words in non-critical regions
## full model
gpt2_non_crit_model <- gam(full_gam, data=rc_rt_prev_non_crit.gpt2.data)
summary(gpt2_non_crit_model)
gam.check(gpt2_non_crit_model)

## base model
gpt2_non_crit_base_model <- gam(baseline_gam, data=rc_rt_prev_non_crit.gpt2.data)
summary(gpt2_non_crit_base_model)
gam.check(gpt2_non_crit_base_model)

## model comparison on the training part
gpt2_non_crit_ll <- logLik.gam(gpt2_non_crit_model)
gpt2_non_crit_base_ll <- logLik.gam(gpt2_non_crit_base_model)
gpt2_non_crit_ll - gpt2_non_crit_base_ll

## predict the surprisal of words in critical regions using the full gam model 
gpt2_crit_predictions <- predict(gpt2_non_crit_model, newdata=rc_rt_prev_crit.gpt2.data,type="response", se.fit=TRUE)
ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.gpt2.data$mean_rt,
                       predicted_rt=gpt2_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1,color="black")+
  labs(x="Actual reading time (critical region)",
       y="Predicted reading time")

# compute rmse
rmse_gpt2 <- sqrt(mean((gpt2_crit_predictions$fit - rc_rt_prev_crit.gpt2.data$mean_rt)^2))
rmse_gpt2 # 92.22555

# compute the delta log-likelihood
gpt2_crit_base_ll <- get_loglikelihood(gpt2_non_crit_base_model, rc_rt_prev_crit.gpt2.data, rc_rt_prev_crit.gpt2.data$mean_rt)
gpt2_crit_ll <- get_loglikelihood(gpt2_non_crit_model, rc_rt_prev_crit.gpt2.data, rc_rt_prev_crit.gpt2.data$mean_rt)
gpt2_crit_delta_ll <- gpt2_crit_ll - gpt2_crit_base_ll
gpt2_crit_delta_ll

### Llama-3.2-1B ----
#### predicting using all words ----
rc_rt_prev.llama1B.data <- get_prev(rc_rt.llama1B.data,2)

llama1B_model <- gam(full_gam,data=rc_rt_prev.llama1B.data)
summary(llama1B_model)
gam.check(llama1B_model)

## model comparison 
llama1B_gam_ll <- logLik.gam(llama1B_model)
llama1B_gam_delta_ll <- llama1B_gam_ll - gam_base_ll
llama1B_gam_delta_ll # 31.74

## cross-validation
llama1B_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.llama1B.data,k=10)
rmse_llama1B_gam_cv <- llama1B_gam_cv$rmse_vec
delta_ll_llama1B_gam_cv <- llama1B_gam_cv$delta_ll
delta_ll_llama1B_gam_cv

# for visualization purpose
## create the dataframe for prediction and ploting
llama1B_prob_new_data <- get_new_df(rc_rt_prev.llama1B.data, 100, 2)
## predict using the new dataframe
llama1B_predictions <- predict(llama1B_model, newdata=llama1B_prob_new_data,type="response",
                               se.fit=TRUE)
## plot predictions
llama1B_rt_graph <- plot_predictions(llama1B_prob_new_data, llama1B_predictions) 
llama1B_rt_graph

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.llama1B.data <- rc_rt.llama1B.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.llama1B.data <- get_prev(rc_rt.llama1B.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))
## full model
llama1B_non_crit_model <- gam(full_gam, data=rc_rt_prev_non_crit.llama1B.data)
summary(llama1B_non_crit_model)
gam.check(llama1B_non_crit_model)

## base model
llama1B_non_crit_base_model <- gam(baseline_gam, data=rc_rt_prev_non_crit.llama1B.data)
summary(llama1B_non_crit_base_model)
gam.check(llama1B_non_crit_base_model)

## model comparison on the training part
llama1B_non_crit_ll <- logLik.gam(llama1B_non_crit_model)
llama1B_non_crit_base_ll <- logLik.gam(llama1B_non_crit_base_model)
llama1B_non_crit_ll - llama1B_non_crit_base_ll

## predict the surprisal of words in critical regions using the full gam model 
llama1B_crit_predictions <- predict(llama1B_non_crit_model, newdata=rc_rt_prev_crit.llama1B.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.llama1B.data$mean_rt,
                       predicted_rt=llama1B_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1,color="black")+
  labs(x="Actual reading time",
       y="Predicted reading time")

# compute rmse
rmse_llama1B <- sqrt(mean((llama1B_crit_predictions$fit - rc_rt_prev_crit.llama1B.data$mean_rt)^2))
rmse_llama1B  # 90.275

# compute the log likelihood
llama1B_crit_base_ll <- get_loglikelihood(llama1B_non_crit_base_model, rc_rt_prev_crit.llama1B.data, rc_rt_prev_crit.llama1B.data$mean_rt)
llama1B_crit_ll <- get_loglikelihood(llama1B_non_crit_model, rc_rt_prev_crit.llama1B.data, rc_rt_prev_crit.llama1B.data$mean_rt)
llama1B_crit_delta_ll <- llama1B_crit_ll - llama1B_crit_base_ll
llama1B_crit_delta_ll

### Llama3.2-3B ----
rc_rt.llama3B.data <- rc_rt.llama3B.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                  "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.llama3B.data <- get_prev(rc_rt.llama3B.data, 2)

llama3B_model <- gam(full_gam, data=rc_rt_prev.llama3B.data)
summary(llama3B_model)
gam.check(llama3B_model)

## model comparison 
llama3B_gam_ll <- logLik.gam(llama3B_model)
llama3B_gam_delta_ll <- llama3B_gam_ll - gam_base_ll
llama3B_gam_delta_ll # 13.69072

## cross-validation
llama3B_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.llama3B.data,k=10)
rmse_llama3B_gam_cv <- llama3B_gam_cv$rmse_vec
delta_ll_llama3B_gam_cv <- llama3B_gam_cv$delta_ll
delta_ll_llama3B_gam_cv

# for visualization purpose
## create the dataframe for prediction and ploting
llama3B_prob_new_data <- get_new_df(rc_rt_prev.llama3B.data, 100, 2)
## predict using the new dataframe
llama3B_predictions <- predict(llama3B_model, newdata=llama3B_prob_new_data,type="response",
                               se.fit=TRUE)
## plot the predictions
llama3B_rt_graph <- plot_predictions(llama3B_prob_new_data,llama3B_predictions)
llama3B_rt_graph

ggplot(data=data.frame(surprisal=rc_rt_prev.llama3B.data$surprisal,
                       actual_rt=rc_rt_prev.llama3B.data$mean_rt),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="surprisal",
       y="Reading time")

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.llama3B.data <- rc_rt.llama3B.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.llama3B.data <- get_prev(rc_rt.llama3B.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit the gam model on words in non-critical regions
## full model
llama3B_non_crit_model <- gam(full_gam, data=rc_rt_prev_non_crit.llama3B.data)
summary(llama3B_non_crit_model)
gam.check(llama3B_non_crit_model)

## base model
llama3B_non_crit_base_model <- gam(baseline_gam, data=rc_rt_prev_non_crit.llama3B.data)
summary(llama3B_non_crit_base_model)
gam.check(llama3B_non_crit_base_model)

## model comparison on the training part
llama3B_non_crit_ll <- logLik.gam(llama3B_non_crit_model)
llama3B_non_crit_base_ll <- logLik.gam(llama3B_non_crit_base_model)
llama3B_non_crit_ll - llama3B_non_crit_base_ll

## predict the surprisal of words in critical regions using the full gam model 
llama3B_crit_predictions <- predict(llama3B_non_crit_model, newdata=rc_rt_prev_crit.llama3B.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.llama3B.data$mean_rt,
                       predicted_rt=llama3B_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1,color="black")+
  labs(x="Actual reading time",
       y="Predicted reading time")

# compute rmse
rmse_llama3B <- sqrt(mean((llama3B_crit_predictions$fit - rc_rt_prev_crit.llama3B.data$mean_rt)^2))
rmse_llama3B # 82.48731

# compute the log likelihood
llama3B_crit_base_ll <- get_loglikelihood(llama3B_non_crit_base_model, rc_rt_prev_crit.llama3B.data, rc_rt_prev_crit.llama3B.data$mean_rt)
llama3B_crit_ll <- get_loglikelihood(llama3B_non_crit_model, rc_rt_prev_crit.llama3B.data, rc_rt_prev_crit.llama3B.data$mean_rt)
llama3B_crit_delta_ll <- llama3B_crit_ll - llama3B_crit_base_ll
llama3B_crit_delta_ll

### Llama3.2-1B-Instruct ----
rc_rt.llama1B_instruct.data <- rc_rt.llama1B_instruct.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                  "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.llama1B_instruct.data <- get_prev(rc_rt.llama1B_instruct.data, 2)

llama1B_instruct_model <- gam(full_gam, data = rc_rt_prev.llama1B_instruct.data)
summary(llama1B_instruct_model)
gam.check(llama1B_instruct_model)

## model comparison 
llama1B_instruct_gam_ll <- logLik.gam(llama1B_instruct_model)
llama1B_instruct_gam_delta_ll <- llama1B_instruct_gam_ll - gam_base_ll
llama1B_instruct_gam_delta_ll # 10.8884

## cross-validation
llama1B_instruct_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.llama1B_instruct.data,k=10)
rmse_llama1B_instruct_gam_cv <- llama1B_instruct_gam_cv$rmse_vec
delta_ll_llama1B_instruct_gam_cv <- llama1B_instruct_gam_cv$delta_ll
delta_ll_llama1B_instruct_gam_cv

# for visualization purpose
## create the dataframe for prediction and ploting
llama1B_instruct_prob_new_data <- get_new_df(rc_rt_prev.llama1B_instruct.data, 100, 2)
## predict using the new dataframe
llama1B_instruct_predictions <- predict(llama1B_instruct_model, newdata=llama1B_instruct_prob_new_data,type="response",
                                        se.fit=TRUE)
## plot the predictions
llama1B_instruct_rt_graph <- plot_predictions(llama1B_instruct_prob_new_data,llama1B_instruct_predictions)
llama1B_instruct_rt_graph

ggplot(data=data.frame(surprisal=rc_rt_prev.llama1B_instruct.data$surprisal,
                       actual_rt=rc_rt_prev.llama1B_instruct.data$mean_rt),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="surprisal",
       y="Reading time")

## cross-validation
llama1B_instruct_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.llama1B_instruct.data)
rmse_llama1B_instruct_gam_cv <- llama1B_instruct_gam_cv$rmse_vec
delta_ll_llama1B_instruct_gam_cv <- llama1B_instruct_gam_cv$delta_ll

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.llama1B_instruct.data <- rc_rt.llama1B_instruct.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.llama1B_instruct.data <- get_prev(rc_rt.llama1B_instruct.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit the gam model on words in non-critical regions
## full model
llama1B_instruct_non_crit_model <- gam(full_gam, data = rc_rt_prev_non_crit.llama1B_instruct.data)
summary(llama1B_instruct_non_crit_model)
gam.check(llama1B_instruct_non_crit_model)

## base model
llama1B_instruct_non_crit_base_model <- gam(baseline_gam,
                                            data=rc_rt_prev_non_crit.llama1B_instruct.data)
summary(llama1B_instruct_non_crit_base_model)
gam.check(llama1B_instruct_non_crit_base_model)

## model comparison on the training part
llama1B_instruct_non_crit_ll <- logLik.gam(llama1B_instruct_non_crit_model)
llama1B_instruct_non_crit_base_ll <- logLik.gam(llama1B_instruct_non_crit_base_model)
llama1B_instruct_non_crit_ll - llama1B_instruct_non_crit_base_ll

## predict the surprisal of words in critical regions using the full gam model 
llama1B_instruct_crit_predictions <- predict(llama1B_instruct_non_crit_model, 
                                             newdata=rc_rt_prev_crit.llama1B_instruct.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.llama1B_instruct.data$mean_rt,
                       predicted_rt=llama1B_instruct_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1,color="black")+
  labs(x="Actual reading time",
       y="Predicted reading time")

# compute rmse
rmse_llama1B_instruct <- sqrt(mean((llama1B_instruct_crit_predictions$fit - rc_rt_prev_crit.llama1B_instruct.data$mean_rt)^2))
rmse_llama1B_instruct # 86.69788

# compute delta log likelihood
llama1B_instruct_crit_base_ll <- get_loglikelihood(llama1B_instruct_non_crit_base_model, rc_rt_prev_crit.llama1B_instruct.data, rc_rt_prev_crit.llama1B_instruct.data$mean_rt)
llama1B_instruct_crit_ll <- get_loglikelihood(llama1B_instruct_non_crit_model, rc_rt_prev_crit.llama1B_instruct.data, rc_rt_prev_crit.llama1B_instruct.data$mean_rt)
llama1B_instruct_crit_delta_ll <- llama1B_instruct_crit_ll - llama1B_instruct_crit_base_ll
llama1B_instruct_crit_delta_ll

### Llama3.2-3B-Instruct ----
rc_rt.llama3B_instruct.data <- rc_rt.llama3B_instruct.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3",
                                  "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.llama3B_instruct.data <- get_prev(rc_rt.llama3B_instruct.data, 2)

llama3B_instruct_model <- gam(full_gam,data=rc_rt_prev.llama3B_instruct.data)
summary(llama3B_instruct_model)
gam.check(llama3B_instruct_model)

## model comparison 
llama3B_instruct_gam_ll <- logLik.gam(llama3B_instruct_model)
llama3B_instruct_gam_delta_ll <- llama3B_instruct_gam_ll - gam_base_ll
llama3B_instruct_gam_delta_ll # 19.63077

## cross-validation
llama3B_instruct_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.llama3B_instruct.data,k=10)
rmse_llama3B_instruct_gam_cv <- llama3B_instruct_gam_cv$rmse_vec
delta_ll_llama3B_instruct_gam_cv <- llama3B_instruct_gam_cv$delta_ll
delta_ll_llama3B_instruct_gam_cv

# for visualization purpose
## create the dataframe for prediction and ploting
llama3B_instruct_prob_new_data <- get_new_df(rc_rt_prev.llama3B_instruct.data, 100, 2)
## predict using the new dataframe
llama3B_instruct_predictions <- predict(llama3B_instruct_model, newdata=llama3B_instruct_prob_new_data,type="response",
                                        se.fit=TRUE)
## plot the predictions
llama3B_instruct_rt_graph <- plot_predictions(llama3B_instruct_prob_new_data,llama3B_instruct_predictions)
llama3B_instruct_rt_graph

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.llama3B_instruct.data <- rc_rt.llama3B_instruct.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.llama3B_instruct.data <- get_prev(rc_rt.llama3B_instruct.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit the gam model on words in non-critical regions
## full model
llama3B_instruct_non_crit_model <- gam(full_gam, 
                                       data = rc_rt_prev_non_crit.llama3B_instruct.data)
summary(llama3B_instruct_non_crit_model)
gam.check(llama3B_instruct_non_crit_model)

## base model
llama3B_instruct_non_crit_base_model <- gam(baseline_gam,
                                            data=rc_rt_prev_non_crit.llama3B_instruct.data)
summary(llama3B_instruct_non_crit_base_model)
gam.check(llama3B_instruct_non_crit_base_model)

## model comparison on the training part
llama3B_instruct_non_crit_ll <- logLik.gam(llama3B_instruct_non_crit_model)
llama3B_instruct_non_crit_base_ll <- logLik.gam(llama3B_instruct_non_crit_base_model)
llama3B_instruct_non_crit_ll - llama3B_instruct_non_crit_base_ll

## predict the surprisal of words in critical regions using the full gam model 
llama3B_instruct_crit_predictions <- predict(llama3B_instruct_non_crit_model, 
                                             newdata=rc_rt_prev_crit.llama3B_instruct.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.llama3B_instruct.data$mean_rt,
                       predicted_rt=llama3B_instruct_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1,color="black")+
  labs(x="Actual reading time",
       y="Predicted reading time")

# compute rmse
rmse_llama3B_instruct <- sqrt(mean((llama3B_instruct_crit_predictions$fit - rc_rt_prev_crit.llama3B_instruct.data$mean_rt)^2))
rmse_llama3B_instruct # 92.0332

# compute the log likelihood
llama3B_instruct_crit_base_ll <- get_loglikelihood(llama3B_instruct_non_crit_base_model, rc_rt_prev_crit.llama3B_instruct.data, rc_rt_prev_crit.llama3B_instruct.data$mean_rt)
llama3B_instruct_crit_ll <- get_loglikelihood(llama3B_instruct_non_crit_model, rc_rt_prev_crit.llama3B_instruct.data, rc_rt_prev_crit.llama3B_instruct.data$mean_rt)
llama3B_instruct_crit_delta_ll <- llama3B_instruct_crit_ll - llama3B_instruct_crit_base_ll
llama3B_instruct_crit_delta_ll

### All model summary ----
model <- c("GPT2", "1B", "1B-Instruct", "3B", "3B-Instruct")
perplexity <- c(gpt2_perplexity, llama1B_perplexity, llama1B_instruct_perplexity, llama3B_perplexity, llama3B_instruct_perplexity)

gam_delta_ll <- c(gpt2_gam_delta_ll, llama1B_gam_delta_ll, llama1B_instruct_gam_delta_ll, llama3B_gam_delta_ll, llama3B_instruct_gam_delta_ll)
gam_delta_ll.data <- data.frame(model=model, delta_ll = gam_delta_ll, perplexity = perplexity) %>% 
  arrange(model, delta_ll)

gam_delta_ll_graph <- ggplot(gam_delta_ll.data %>% 
                               filter(model != "pythia1b"),
                             aes(x=perplexity,
                                 y=delta_ll)) + 
  geom_smooth(method="lm", formula=y~x,se=TRUE, color="black") +
  geom_point(aes(color=model),size=5)+
  labs(x="Perplexity",
       y="ΔLogLik") +
scale_color_manual(values=cbPalette) +
  theme(legend.text = element_text(size=12), 
        legend.position = "top",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 12),
        axis.title.y = element_text(size = 12))
gam_delta_ll_graph

#### rmse ----
# order should be gpt2, llama1b, llama1b-instruct, llama3b, llama3b-instruct
rmse <- c(rmse_gpt2, rmse_llama1B, rmse_llama1B_instruct, rmse_llama3B, rmse_llama3B_instruct)
gam_rmse.data <- data.frame(model=model, rmse = rmse, perplexity = perplexity) %>% 
  arrange(model, rmse)
gam_rmse_summary <- gam_rmse.data %>% 
  group_by(model, perplexity) %>% 
  summarize(Mean = mean(rmse),
            CILow = ci.low(rmse),
            CIHigh = ci.high(rmse)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)
gam_rmse_graph <- ggplot(gam_rmse_summary,
                                aes(x=perplexity,
                                    y=Mean)) +
  geom_smooth(method="lm", formula=y~x,se=TRUE, color="black") +
  geom_point(aes(color=model),size=5)+
  labs(x="Perplexity",
       y="RMSE") +
  scale_color_manual(values=cbPalette) +
  theme(legend.text = element_text(size=12),
        legend.position = "top",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 12),
        axis.title.y = element_text(size = 12))
gam_rmse_graph

#### cross-validation ----
# order should be gpt2, llama1b, llama1b-instruct, llama3b, llama3b-instruct
gam_delta_ll_cv <- c(delta_ll_gpt2_gam_cv, delta_ll_llama1B_gam_cv, delta_ll_llama1B_instruct_gam_cv, delta_ll_llama3B_gam_cv, delta_ll_llama3B_instruct_gam_cv)
gam_delta_ll_cv.data <- data.frame(model=model, delta_ll = gam_delta_ll_cv, perplexity = perplexity) %>% 
  arrange(model, delta_ll)
gam_delta_ll_cv_summary <- gam_delta_ll_cv.data %>% 
  group_by(model, perplexity) %>% 
  summarize(Mean = mean(delta_ll),
            CILow = ci.low(delta_ll),
            CIHigh = ci.high(delta_ll)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

gam_delta_ll_cv_graph <- ggplot(gam_delta_ll_cv_summary,
                                aes(x=perplexity,
                                    y=Mean)) +
  geom_smooth(method="lm", formula=y~x,se=TRUE, color="black") +
  geom_point(aes(color=model),size=5)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.1, 
                show.legend = FALSE) +
  labs(x="Perplexity",
       y="ΔLogLik") +
  scale_color_manual(values=cbPalette) +
  theme(legend.text = element_text(size=10))
gam_delta_ll_cv_graph
