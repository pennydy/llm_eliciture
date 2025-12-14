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
frequency.data <- read.csv("../../../data/eliciture_frequency.csv", header=TRUE) %>% 
  select(-c(X, X.1))

## GPT-2 ----
rc_rt.gpt2.data <- read.csv("../../../data/rc_rt/rc_rt_combined_gpt2_3.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.gpt2.data <- left_join(rc_rt.gpt2.data, frequency.data, by=c("word","frequency","log_freq")) %>% 
  select(-c("frequency", "log_freq", "surp1",  "word1", "word2", "surp2",  "frequency2", "word3", "surp3", "frequency3")) %>% 
  rename(frequency="frequency1",
         log_freq="log_freq1")

# perplexity based on the stimuli in this exp
gpt2_perplexity <- 5.536873862777304
# perplexity based on natural stories (futrell et al. 2021) for sanity check
# gpt2_perplexity <- 4.359453969447
  
rc_rt.gpt2.summary <- rc_rt.gpt2.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
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
  left_join(frequency.data, by=c("word","frequency","log_freq")) %>% 
  select(-c("frequency", "log_freq", "surp1",  "word1", "word2", "surp2",  "frequency2", "word3", "surp3", "frequency3")) %>% 
  rename(frequency="frequency1",
         log_freq="log_freq1")

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

rc_rt.llama3B.data <- left_join(rc_rt.llama3B.data, frequency.data, by=c("word","frequency","log_freq")) %>% 
  select(-c("frequency", "log_freq", "surp1",  "word1", "word2", "surp2",  "frequency2", "word3", "surp3", "frequency3")) %>%
  rename(frequency="frequency1",
         log_freq="log_freq1")

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

rc_rt.llama1B_instruct.data <- left_join(rc_rt.llama1B_instruct.data, frequency.data, by=c("word","frequency","log_freq")) %>% 
  select(-c("frequency", "log_freq", "surp1",  "word1", "word2", "surp2",  "frequency2", "word3", "surp3", "frequency3")) %>%
  rename(frequency="frequency1",
         log_freq="log_freq1")

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

rc_rt.llama3B_instruct.data <- left_join(rc_rt.llama3B_instruct.data, frequency.data, by=c("word","frequency","log_freq")) %>% 
  select(-c("frequency", "log_freq", "surp1",  "word1", "word2", "surp2",  "frequency2", "word3", "surp3", "frequency3")) %>%
  rename(frequency="frequency1",
         log_freq="log_freq1")

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

## Pythia1B ----
rc_rt.pythia1B.data <- read.csv("../../../data/rc_rt/rc_rt_combined_pythia-1b_2.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.pythia1B.data <- left_join(rc_rt.pythia1B.data, frequency.data, by=c("word","frequency","log_freq"))%>% 
  select(-c("frequency", "log_freq", "surp1",  "word1", "word2", "surp2",  "frequency2", "word3", "surp3", "frequency3")) %>%
  rename(frequency="frequency1",
         log_freq="log_freq1")

pythia1B_perplexity <- rc_rt.pythia1B.data %>% 
  summarize(mean=mean(surprisal),
            perplexity=exp(mean)) %>% 
  pull(perplexity)
pythia1B_perplexity

rc_rt.pythia1B.summary <- rc_rt.pythia1B.data %>% 
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

# 2. Plot ----
## GPT2 ----
gpt2_graph <- ggplot(rc_rt.gpt2.summary,
                     aes(x=crit,
                         y=Mean,
                         group=cond,
                         linetype=cond,
                         color=cond)) +
  geom_point() +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  geom_line() +
  labs(y="Mean suprisal (bits)",
       x="Critical region") +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust=1)) +
  scale_color_brewer(palette = "Dark2")
gpt2_graph
ggsave(gpt2_graph, file="../graphs/rc_rt_gpt2.pdf", width=8, height=4)

## Llama3.2-1B ----
llama1B_graph <- ggplot(rc_rt.llama1B.summary,
                          aes(x=crit,
                              y=Mean,
                              group=cond,
                              linetype=cond,
                              color=cond)) +
  geom_point() +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  geom_line() +
  labs(y="Mean suprisal (bits)",
       x="Critical region") +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust=1)) +
  scale_color_brewer(palette = "Dark2")
llama1B_graph
ggsave(llama1B_graph, file="../graphs/rc_rt_llama1B.pdf", width=8, height=4)

## Llama3.2-3B ----
llama3B_graph <- ggplot(rc_rt.llama3B.summary,
                        aes(x=crit,
                            y=Mean,
                            group=cond,
                            linetype=cond,
                            color=cond)) +
  geom_point() +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  geom_line() +
  labs(y="Mean suprisal (bits)",
       x="Critical region") +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust=1)) +
  scale_color_brewer(palette = "Dark2")
llama3B_graph
ggsave(llama3B_graph, file="../graphs/rc_rt_llama3B.pdf", width=8, height=4)

## Llama3.2-1B-Instruct ----
llama1B_instruct_graph <- ggplot(rc_rt.llama1B_instruct.summary,
                                 aes(x=crit,
                                     y=Mean,
                                     group=cond,
                                     linetype=cond,
                                     color=cond)) +
  geom_point() +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  geom_line() +
  labs(y="Mean suprisal (bits)",
       x="Critical region") +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust=1)) +
  scale_color_brewer(palette = "Dark2")
llama1B_instruct_graph
ggsave(llama1B_instruct_graph, file="../graphs/rc_rt_llama1B_instruct.pdf", width=8, height=4)

## Llama3.2-3B-Instruct ----
llama3B_instruct_graph <- ggplot(rc_rt.llama3B_instruct.summary,
                                 aes(x=crit,
                                     y=Mean,
                                     group=cond,
                                     linetype=cond,
                                     color=cond)) +
  geom_point() +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  geom_line() +
  labs(y="Mean suprisal (bits)",
       x="Critical region") +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust=1)) +
  scale_color_brewer(palette = "Dark2")
llama3B_instruct_graph
ggsave(llama3B_instruct_graph, file="../graphs/rc_rt_llama3B_instruct.pdf", width=8, height=4)

## Pythia1B ----
pythia1B_graph <- ggplot(rc_rt.pythia1B.summary,
                                 aes(x=crit,
                                     y=Mean,
                                     group=cond,
                                     linetype=cond,
                                     color=cond)) +
  geom_point() +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  geom_line() +
  labs(y="Mean suprisal (bits)",
       x="Critical region") +
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust=1)) +
  scale_color_brewer(palette = "Dark2")
pythia1B_graph
ggsave(pythia1B_graph, file="../graphs/rc_rt_pythia1B.pdf", width=8, height=4)

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
           prev1_freq = lag(log_freq, order_by = crit),
           prev1_surpsum = lag(surp_sum, order_by = crit),
           
           prev2_word = lag(prev1_word, order_by = crit),
           prev2_wordlen = lag(prev1_wordlen, order_by = crit),
           prev2_surp = lag(prev1_surp, order_by = crit),
           prev2_freq = lag(prev1_freq, order_by = crit),
           prev2_surpsum = lag(prev1_surpsum, order_by = crit))
  if(num_prev==3) { # if getting the measures of the previous three words 
    prev_df <- prev_df %>% 
      mutate(prev3_word = lag(prev2_word, order_by = crit),
             prev3_wordlen = lag(prev2_wordlen, order_by = crit),
             prev3_surp = lag(prev2_surp, order_by = crit),
             prev3_freq = lag(prev2_freq, order_by = crit),
             prev3_surpsum = lag(prev2_surpsum, order_by = crit))  %>%
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
                       log_freq = seq(min(dataframe$log_freq),
                                  max(dataframe$log_freq),
                                  length.out = length_out),
                       prev1_freq = seq(min(dataframe$prev1_freq),
                                       max(dataframe$prev1_freq),
                                       length.out = length_out),
                       prev2_freq = seq(min(dataframe$prev2_freq),
                                        max(dataframe$prev2_freq),
                                        length.out = length_out),
                       surp_sum = seq(min(dataframe$surp_sum),
                                      max(dataframe$surp_sum),
                                      length.out = length_out),
                       prev1_surpsum = seq(min(dataframe$prev1_surpsum),
                                           max(dataframe$prev1_surpsum),
                                           length.out = length_out),
                       prev2_surpsum = seq(min(dataframe$prev2_surpsum),
                                           max(dataframe$prev2_surpsum),
                                           length.out = length_out))
  if(num_prev==3) {
    new_df <- new_df %>% 
      mutate(prev3_surp = seq(min(dataframe$prev3_surp),
                              max(dataframe$prev3_surp),
                              length.out = length_out), 
             prev3_wordlen = seq(min(dataframe$prev3_wordlen),
                                 max(dataframe$prev3_wordlen),
                                 length.out=length_out),
             prev3_freq = seq(min(dataframe$prev3_freq),
                              max(dataframe$prev3_freq),
                              length.out = length_out),
             prev3_surpsum = seq(min(dataframe$prev3_surpsum),
                                 max(dataframe$prev3_surpsum),
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
baseline_prev3_gam = mean_rt ~ te(wordlen, bs="cr") +
  te(prev1_wordlen, bs = "cr") +
  te(prev2_wordlen, bs = "cr") + 
  te(prev3_wordlen, bs = "cr") +
  te(surp_sum, bs = "cr") +
  te(prev1_surpsum, bs = "cr") +
  te(prev2_surpsum, bs = "cr") +
  te(prev3_surpsum, bs = "cr")
  # te(log_freq, bs = "cr") + 
  # te(prev1_freq, bs = "cr") +
  # te(prev2_freq, bs = "cr") +
  # te(prev3_freq, bs = "cr")
full_prev3_gam = mean_rt ~ s(surprisal, bs="cr", k=5) + 
  s(prev1_surp, bs="cr", k=5) + 
  s(prev2_surp, bs="cr", k=5) +
  s(prev3_surp, bs="cr",k=5) +
  te(wordlen, bs="cr") + 
  te(prev1_wordlen, bs="cr") + 
  te(prev2_wordlen, bs="cr") + 
  te(prev3_wordlen, bs="cr") +
  # te(log_freq, bs="cr") +
  # te(prev1_freq, bs="cr") +
  # te(prev2_freq, bs="cr") +
  # te(prev3_freq, bs="cr")
  te(surp_sum, bs = "cr") +
  te(prev1_surpsum, bs = "cr") +
  te(prev2_surpsum, bs = "cr") +
  te(prev3_surpsum, bs = "cr")

baseline_gam = mean_rt ~ te(wordlen, bs="cr") +
  te(prev1_wordlen, bs = "cr") +
  te(prev2_wordlen, bs = "cr") + 
  te(surp_sum, bs = "cr") +
  te(prev1_surpsum, bs = "cr") +
  te(prev2_surpsum, bs = "cr")
  # te(log_freq, bs = "cr") + # use surpsum instead of log freq
  # te(prev1_freq, bs = "cr") +
  # te(prev2_freq, bs = "cr")
full_gam = mean_rt ~ s(surprisal, bs="cr", k=5) + 
  s(prev1_surp, bs="cr", k=5) + 
  s(prev2_surp, bs="cr", k=5) +
  te(wordlen, bs="cr") + 
  te(prev1_wordlen, bs="cr") + 
  te(prev2_wordlen, bs="cr") + 
  te(surp_sum, bs="cr") +
  te(prev1_surpsum, bs = "cr") +
  te(prev2_surpsum, bs = "cr")
  # te(log_freq, bs="cr") + # use surpsum instead of log freq
  # te(prev1_freq, bs="cr") +
  # te(prev2_freq, bs="cr")

loglik_testset <- function(y_obs, y_pred, sigma) {
  sum(dnorm(y_obs, mean = y_pred, sd = sigma, log = TRUE))
}

cv_gam <- function(full_formula, simple_formula, data, k=10) {
  set.seed(1024)
  n <- nrow(data)
  folds <- sample(rep(1:k, length.out = n))
  rmse_vec <- numeric(k)
  delta_ll <- numeric(k)
  
  for (i in 1:k) {
    train <- data[folds != i, ]
    test <- data[folds == i, ]
    full_model <- gam(full_formula, data=train)
    full_preds <- predict(full_model, newdata=test)
    full_sigma <- sqrt(mean(residuals(full_model)^2))
    full_loglike <- loglik_testset(test$mean_rt, full_preds, full_sigma)
    
    simple_model <- gam(simple_formula, data=train)
    simple_preds <- predict(simple_model, newdata=test)
    simple_sigma <- sqrt(mean(residuals(simple_model)^2))
    simple_loglike <- loglik_testset(test$mean_rt, simple_preds, simple_sigma)
    
    delta_ll[i] <- full_loglike - simple_loglike 
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
gpt2_gam_delta_ll

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

# plot predicted rt against surprisal <-- not meaningful
ggplot(data=data.frame(surprisal=rc_rt_prev.gpt2.data$surprisal,
                       actual_rt=rc_rt_prev.gpt2.data$mean_rt),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="surprisal",
       y="Reading time")

## cross-validation
# train_control <- trainControl(method = "cv", number=10)
# gpt2_gam_cv <- train(mean_rt ~ surprisal + prev1_surp + prev2_surp + wordlen + prev1_wordlen + prev2_wordlen + surp_sum + prev1_surpsum + prev2_surpsum,
#                      data=rc_rt_prev.gpt2.data,
#                      method="gam",
#                      trControl = train_control
# )
gpt2_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.gpt2.data)
rmse_gpt2_gam_cv <- gpt2_gam_cv$rmse_vec
delta_ll_gpt2_gam_cv <- gpt2_gam_cv$delta_ll

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

# for visualization purpose -- non-critical regions
## create dataframe with equal entries across columns
gpt2_non_crit_new_data <- get_new_df(rc_rt_prev_non_crit.gpt2.data, 100, 2)

## predict the surprisal of words in non-critical regions using the full gam model
gpt2_non_crit_predictions <- predict(gpt2_non_crit_model, newdata=gpt2_non_crit_new_data,type="response",
                            se.fit=TRUE)
## plot predictions (non-critical regions)
plot_predictions(gpt2_non_crit_new_data, gpt2_non_crit_predictions)

## predict the surprisal of words in non-critical regions using the base gam model
gpt2_non_crit_base_predictions <- predict(gpt2_non_crit_base_model, newdata=gpt2_non_crit_new_data,type="response",
                                     se.fit=TRUE)
## plot base model (non-critical regions)
plot_predictions(gpt2_non_crit_new_data, gpt2_non_crit_base_predictions)

# for visualization purpose -- critical regions
## create dataframe with equal entries across columns
gpt2_crit_new_data <- get_new_df(rc_rt_prev_crit.gpt2.data, 100, 2)

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
  labs(x="Actual reading time",
       y="Predicted reading time")

# compute rmse
rmse_gpt2 <- sqrt(mean((gpt2_crit_predictions$fit - rc_rt_prev_crit.gpt2.data$mean_rt)^2))
rmse_gpt2

## plot full model (critical regions)
gpt2_crit_predictions <- predict(gpt2_non_crit_model, newdata=gpt2_crit_new_data,type="response", se.fit=TRUE)
gpt2_crit_graph <- plot_predictions(gpt2_crit_new_data, gpt2_crit_predictions)
gpt2_crit_graph
ggsave(gpt2_crit_graph, file="../graphs/gpt2_crit_graph.pdf", width=8, height=4)

## predict the surprisal of words in critical regions using the full gam model 
gpt2_crit_base_predictions <- predict(gpt2_non_crit_base_model, newdata=gpt2_crit_new_data,type="response",
                                 se.fit=TRUE)
## plot base model (critical regions)
plot_predictions(gpt2_crit_new_data, gpt2_crit_base_predictions)

# compute the delta log-likelihood
gpt2_crit_base_ll <- get_loglikelihood(gpt2_non_crit_base_model, rc_rt_prev_crit.gpt2.data, rc_rt_prev_crit.gpt2.data$mean_rt)
gpt2_crit_ll <- get_loglikelihood(gpt2_non_crit_model, rc_rt_prev_crit.gpt2.data, rc_rt_prev_crit.gpt2.data$mean_rt)
gpt2_crit_delta_ll <- gpt2_crit_ll - gpt2_crit_base_ll
gpt2_crit_delta_ll


### Llama3.2-1B ----
rc_rt.llama1B.data <- rc_rt.llama1B.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3",
                                  "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.llama1B.data <- get_prev(rc_rt.llama1B.data,2)

llama1B_model <- gam(full_gam,data=rc_rt_prev.llama1B.data)
summary(llama1B_model)
gam.check(llama1B_model)

## model comparison 
llama1B_gam_ll <- logLik.gam(llama1B_model)
llama1B_gam_delta_ll <- llama1B_gam_ll - gam_base_ll
llama1B_gam_delta_ll

# for visualization purpose
## create the dataframe for prediction and ploting
llama1B_prob_new_data <- get_new_df(rc_rt_prev.llama1B.data, 100, 2)
## predict using the new dataframe
llama1B_predictions <- predict(llama1B_model, newdata=llama1B_prob_new_data,type="response",
                            se.fit=TRUE)
## plot predictions
llama1B_rt_graph <- plot_predictions(llama1B_prob_new_data, llama1B_predictions) 
llama1B_rt_graph
ggsave(llama1B_rt_graph, file="../graphs/llama1B_rt_freq_graph.pdf", width=8, height=4)

ggplot(data=data.frame(surprisal=rc_rt_prev.llama1B.data$surprisal,
                       actual_rt=rc_rt_prev.llama1B.data$mean_rt),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="surprisal",
       y="Reading time")

## cross-validation
llama1B_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.llama1B.data)
rmse_llama1B_gam_cv <- llama1B_gam_cv$rmse_vec
delta_ll_llama1B_gam_cv <- llama1B_gam_cv$delta_ll

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.llama1B.data <- rc_rt.llama1B.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.llama1B.data <- get_prev(rc_rt.llama1B.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit the gam model on words in non-critical regions
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

# for visualization purpose -- non-critical regions
## create dataframe with equal entries across columns
llama1B_non_crit_new_data <- get_new_df(rc_rt_prev_non_crit.llama1B.data, 100, 2)

## predict the surprisal of words in non-critical regions using the full gam model 
llama1B_non_crit_predictions <- predict(llama1B_non_crit_model, newdata=llama1B_non_crit_new_data,type="response",
                                     se.fit=TRUE)
## plot predictions
plot_predictions(llama1B_non_crit_new_data, llama1B_non_crit_predictions)

## predict the surprisal of words in non-critical regions using the base gam model 
llama1B_non_crit_base_predictions <- predict(llama1B_non_crit_base_model, newdata=llama1B_non_crit_new_data,type="response",
                                          se.fit=TRUE)
## plot base model 
plot_predictions(llama1B_non_crit_new_data, llama1B_non_crit_base_predictions)

# for visualization purpose -- critical regions
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
rmse_llama1B

## create dataframe with equal entries across columns
llama1B_crit_new_data <- get_new_df(rc_rt_prev_crit.llama1B.data, 100, 2)

## plot full model
llama1B_crit_predictions <- predict(llama1B_non_crit_model, newdata=llama1B_crit_new_data,type="response", se.fit=TRUE)
llama1B_crit_graph <- plot_predictions(llama1B_crit_new_data, llama1B_crit_predictions)
llama1B_crit_graph
ggsave(llama1B_crit_graph, file="../graphs/llama1B_crit_graph.pdf", width=8, height=4)


## predict the surprisal of words in critical regions using the full gam model 
llama1B_crit_base_predictions <- predict(llama1B_non_crit_base_model, newdata=llama1B_crit_new_data,type="response",
                                      se.fit=TRUE)
## plot base model
plot_predictions(llama1B_crit_new_data, llama1B_crit_base_predictions)

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
llama3B_gam_delta_ll

# for visualization purpose
## create the dataframe for prediction and ploting
llama3B_prob_new_data <- get_new_df(rc_rt_prev.llama3B.data, 100, 2)
## predict using the new dataframe
llama3B_predictions <- predict(llama3B_model, newdata=llama3B_prob_new_data,type="response",
                               se.fit=TRUE)
## plot the predictions
llama3B_rt_graph <- plot_predictions(llama3B_prob_new_data,llama3B_predictions)
llama3B_rt_graph
ggsave(llama3B_rt_graph, file="../graphs/llama3B_rt_graph.pdf", width=8, height=4)

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

## cross-validation
llama3B_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.llama3B.data)
rmse_llama3B_gam_cv <- llama3B_gam_cv$rmse_vec
delta_ll_llama3B_gam_cv <- llama3B_gam_cv$delta_ll

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

# for visualization purpose -- non-critical regions
## create dataframe with equal entries across columns
llama3B_non_crit_new_data <- get_new_df(rc_rt_prev_non_crit.llama3B.data, 100, 2)

## predict the surprisal of words in non-critical regions using the full gam model 
llama3B_non_crit_predictions <- predict(llama3B_non_crit_model, newdata=llama3B_non_crit_new_data,type="response",
                                        se.fit=TRUE)
## plot predictions
plot_predictions(llama3B_non_crit_new_data, llama3B_non_crit_predictions)

## predict the surprisal of words in non-critical regions using the base gam model 
llama3B_non_crit_base_predictions <- predict(llama3B_non_crit_base_model, newdata=llama3B_non_crit_new_data,type="response",
                                             se.fit=TRUE)
## plot base model 
plot_predictions(llama3B_non_crit_new_data, llama3B_non_crit_base_predictions)

# for visualization purpose -- critical regions
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
rmse_llama3B


## create dataframe with equal entries across columns
llama3B_crit_new_data <- get_new_df(rc_rt_prev_crit.llama3B.data, 100, 2)

## predict the surprisal of words in critical regions using the full gam model 
llama3B_crit_predictions <- predict(llama3B_non_crit_model, newdata=llama3B_crit_new_data,type="response",
                                    se.fit=TRUE)
## plot full model
llama3B_crit_graph <- plot_predictions(llama3B_crit_new_data, llama3B_crit_predictions)
llama3B_crit_graph
ggsave(llama3B_crit_graph, file="../graphs/llama3B_crit_graph.pdf", width=8, height=4)

## predict the surprisal of words in critical regions using the full gam model 
llama3B_crit_base_predictions <- predict(llama3B_non_crit_base_model, newdata=llama3B_crit_new_data,type="response",
                                         se.fit=TRUE)
## plot base model
plot_predictions(llama3B_crit_new_data, llama3B_crit_base_predictions)

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
llama1B_instruct_gam_delta_ll

# for visualization purpose
## create the dataframe for prediction and ploting
llama1B_instruct_prob_new_data <- get_new_df(rc_rt_prev.llama1B_instruct.data, 100, 2)
## predict using the new dataframe
llama1B_instruct_predictions <- predict(llama1B_instruct_model, newdata=llama1B_instruct_prob_new_data,type="response",
                               se.fit=TRUE)
## plot the predictions
llama1B_instruct_rt_graph <- plot_predictions(llama1B_instruct_prob_new_data,llama1B_instruct_predictions)
llama1B_instruct_rt_graph
ggsave(llama1B_instruct_rt_graph, file="../graphs/llama1B_instruct_rt_graph.pdf", width=8, height=4)

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

# for visualization purpose -- non-critical regions
## create dataframe with equal entries across columns
llama1B_instruct_non_crit_new_data <- get_new_df(rc_rt_prev_non_crit.llama1B_instruct.data, 100, 2)

## predict the surprisal of words in non-critical regions using the full gam model 
llama1B_instruct_non_crit_predictions <- predict(llama1B_instruct_non_crit_model, 
                                                 newdata=llama1B_instruct_non_crit_new_data,
                                                 type="response",
                                                 se.fit=TRUE)
## plot predictions
plot_predictions(llama1B_instruct_non_crit_new_data, llama1B_instruct_non_crit_predictions)

## predict the surprisal of words in non-critical regions using the base gam model 
llama1B_instruct_non_crit_base_predictions <- predict(llama1B_instruct_non_crit_base_model, 
                                                      newdata=llama1B_instruct_non_crit_new_data,
                                                      type="response",
                                                      se.fit=TRUE)
## plot base model 
plot_predictions(llama1B_instruct_non_crit_new_data, llama1B_instruct_non_crit_base_predictions)

# for visualization purpose -- critical regions
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
rmse_llama1B_instruct

## create dataframe with equal entries across columns
llama1B_instruct_crit_new_data <- get_new_df(rc_rt_prev_crit.llama1B_instruct.data, 100, 2)

## predict the surprisal of words in critical regions using the full gam model 
llama1B_instruct_crit_predictions <- predict(llama1B_instruct_non_crit_model, 
                                             newdata=llama1B_instruct_crit_new_data,
                                             type="response",
                                             se.fit=TRUE)
## plot full model
llama1B_instruct_crit_graph <- plot_predictions(llama1B_instruct_crit_new_data, llama1B_instruct_crit_predictions)
llama1B_instruct_crit_graph
ggsave(llama1B_instruct_crit_graph, file="../graphs/llama1B_instruct_crit_graph.pdf", width=8, height=4)


## predict the surprisal of words in critical regions using the full gam model 
llama1B_instruct_crit_base_predictions <- predict(llama1B_instruct_non_crit_base_model, 
                                                  newdata=llama1B_instruct_crit_new_data,
                                                  type="response",
                                                  se.fit=TRUE)
## plot base model
plot_predictions(llama1B_instruct_crit_new_data, llama1B_instruct_crit_base_predictions)

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
llama3B_instruct_gam_delta_ll

# for visualization purpose
## create the dataframe for prediction and ploting
llama3B_instruct_prob_new_data <- get_new_df(rc_rt_prev.llama3B_instruct.data, 100, 2)
## predict using the new dataframe
llama3B_instruct_predictions <- predict(llama3B_instruct_model, newdata=llama3B_instruct_prob_new_data,type="response",
                                        se.fit=TRUE)
## plot the predictions
llama3B_instruct_rt_graph <- plot_predictions(llama3B_instruct_prob_new_data,llama3B_instruct_predictions)
llama3B_instruct_rt_graph
ggsave(llama3B_instruct_rt_graph, file="../graphs/llama3B_instruct_rt_graph.pdf", width=8, height=4)

ggplot(data=data.frame(surprisal=rc_rt_prev.llama3B_instruct.data$surprisal,
                       actual_rt=rc_rt_prev.llama3B_instruct.data$mean_rt),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="surprisal",
       y="Reading time")

## cross-validation
llama3B_instruct_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.llama3B_instruct.data)
rmse_llama3B_instruct_gam_cv <- llama3B_instruct_gam_cv$rmse_vec
delta_ll_llama3B_instruct_gam_cv <- llama3B_instruct_gam_cv$delta_ll

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

# for visualization purpose -- non-critical regions
## create dataframe with equal entries across columns
llama3B_instruct_non_crit_new_data <- get_new_df(rc_rt_prev_non_crit.llama3B_instruct.data, 100, 2)

## predict the surprisal of words in non-critical regions using the full gam model 
llama3B_instruct_non_crit_predictions <- predict(llama1B_instruct_non_crit_model, 
                                                 newdata=llama3B_instruct_non_crit_new_data,
                                                 type="response",
                                                 se.fit=TRUE)
## plot predictions
plot_predictions(llama3B_instruct_non_crit_new_data, llama3B_instruct_non_crit_predictions)

## predict the surprisal of words in non-critical regions using the base gam model 
llama3B_instruct_non_crit_base_predictions <- predict(llama3B_instruct_non_crit_base_model, 
                                                      newdata=llama3B_instruct_non_crit_new_data,
                                                      type="response",
                                                      se.fit=TRUE)
## plot base model 
plot_predictions(llama3B_instruct_non_crit_new_data, llama3B_instruct_non_crit_base_predictions)

# for visualization purpose -- critical regions
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
rmse_llama3B_instruct

## create dataframe with equal entries across columns
llama3B_instruct_crit_new_data <- get_new_df(rc_rt_prev_crit.llama3B_instruct.data, 100, 2)

## predict the surprisal of words in critical regions using the full gam model 
llama3B_instruct_crit_predictions <- predict(llama3B_instruct_non_crit_model, 
                                             newdata=llama3B_instruct_crit_new_data,
                                             type="response",
                                             se.fit=TRUE)
## plot full model
llama3B_instruct_crit_graph <- plot_predictions(llama3B_instruct_crit_new_data, llama3B_instruct_crit_predictions)
llama3B_instruct_crit_graph
ggsave(llama3B_instruct_crit_graph, file="../graphs/llama3B_instruct_crit_graph.pdf", width=8, height=4)


## predict the surprisal of words in critical regions using the full gam model 
llama3B_instruct_crit_base_predictions <- predict(llama3B_instruct_non_crit_base_model, 
                                                  newdata=llama3B_instruct_crit_new_data,
                                                  type="response",
                                                  se.fit=TRUE)
## plot base model
plot_predictions(llama3B_instruct_crit_new_data, llama3B_instruct_crit_base_predictions)

llama3B_instruct_crit_base_ll <- get_loglikelihood(llama3B_instruct_non_crit_base_model, rc_rt_prev_crit.llama3B_instruct.data, rc_rt_prev_crit.llama3B_instruct.data$mean_rt)
llama3B_instruct_crit_ll <- get_loglikelihood(llama3B_instruct_non_crit_model, rc_rt_prev_crit.llama3B_instruct.data, rc_rt_prev_crit.llama3B_instruct.data$mean_rt)
llama3B_instruct_crit_delta_ll <- llama3B_instruct_crit_ll - llama3B_instruct_crit_base_ll
llama3B_instruct_crit_delta_ll


### Pythia1B ----
rc_rt.pythia1B.data <- rc_rt.pythia1B.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3",
                                  "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.pythia1B.data <- get_prev(rc_rt.pythia1B.data,2)

pythia1B_model <- gam(full_gam,
                  data=rc_rt_prev.pythia1B.data)
summary(pythia1B_model)
gam.check(pythia1B_model)

## model comparison 
pythia1B_gam_ll <- logLik.gam(pythia1B_model)
pythia1B_gam_delta_ll <- pythia1B_gam_ll - gam_base_ll
pythia1B_gam_delta_ll

# for visualization purpose
## create the dataframe for prediction and ploting
pythia1B_prob_new_data <- get_new_df(rc_rt_prev.pythia1B.data,100,2)
## predict using the new dataframe
pythia1B_predictions <- predict(pythia1B_model, newdata=pythia1B_prob_new_data,type="response",
                            se.fit=TRUE)
## plot the predictions
pythia1B_rt_graph <- plot_predictions(pythia1B_prob_new_data, pythia1B_predictions)
pythia1B_rt_graph
ggsave(pythia1B_rt_graph, file="../graphs/pythia1B_rt_graph.pdf", width=8, height=4)

ggplot(data=data.frame(surprisal=rc_rt_prev.pythia1B.data$surprisal,
                       actual_rt=rc_rt_prev.pythia1B.data$mean_rt),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="gam", formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="surprisal",
       y="Reading time")

## cross-validation
pythia1B_gam_cv <- cv_gam(full_gam, baseline_gam, rc_rt_prev.pythia1B.data)
rmse_pythia1B_gam_cv <- pythia1B_gam_cv$rmse_vec
delta_ll_pythia1B_gam_cv <- pythia1B_gam_cv$delta_ll

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.pythia1B.data <- rc_rt.pythia1B.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.pythia1B.data <- get_prev(rc_rt.pythia1B.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit the gam model on words in non-critical regions
## full model
pythia1B_non_crit_model <- gam(full_gam,
                           data=rc_rt_prev_non_crit.pythia1B.data)
summary(pythia1B_non_crit_model)
gam.check(pythia1B_non_crit_model)

## base model
pythia1B_non_crit_base_model <- gam(baseline_gam, 
                                data=rc_rt_prev_non_crit.pythia1B.data)
summary(pythia1B_non_crit_base_model)
gam.check(pythia1B_non_crit_base_model)

## model comparison on the training part
pythia1B_non_crit_ll <- logLik.gam(pythia1B_non_crit_model)
pythia1B_non_crit_base_ll <- logLik.gam(pythia1B_non_crit_base_model)
pythia1B_non_crit_ll - pythia1B_non_crit_base_ll

# for visualization purpose -- non-critical regions
## create dataframe with equal entries across columns
pythia1B_non_crit_new_data <- get_new_df(rc_rt_prev_non_crit.pythia1B.data, 100, 2)

## predict the surprisal of words in non-critical regions using the full gam model 
pythia1B_non_crit_predictions <- predict(pythia1B_non_crit_model, newdata=pythia1B_non_crit_new_data,type="response",
                                     se.fit=TRUE)
## plot predictions
plot_predictions(pythia1B_non_crit_new_data, pythia1B_non_crit_predictions)

## predict the surprisal of words in non-critical regions using the base gam model 
pythia1B_non_crit_base_predictions <- predict(pythia1B_non_crit_base_model, newdata=pythia1B_non_crit_new_data,type="response",
                                          se.fit=TRUE)
## plot base model 
plot_predictions(pythia1B_non_crit_new_data, pythia1B_non_crit_base_predictions)

# for visualization purpose -- critical regions
## create dataframe with equal entries across columns
pythia1B_crit_new_data <- get_new_df(rc_rt_prev_crit.pythia1B.data, 100, 2)

## predict the surprisal of words in critical regions using the full gam model 
pythia1B_crit_predictions <- predict(pythia1B_non_crit_model, newdata=pythia1B_crit_new_data,type="response",
                                 se.fit=TRUE)
## plot full model
pythia1B_crit_graph <- plot_predictions(pythia1B_crit_new_data, pythia1B_crit_predictions)
pythia1B_crit_graph
ggsave(pythia1B_crit_graph, file="../graphs/pythia1B_crit_graph.pdf", width=8, height=4)

## predict the surprisal of words in critical regions using the full gam model 
pythia1B_crit_base_predictions <- predict(pythia1B_non_crit_base_model, newdata=pythia1B_crit_new_data,type="response",
                                      se.fit=TRUE)
## plot base model
plot_predictions(pythia1B_crit_new_data, pythia1B_crit_base_predictions)

# compute the delta log-likelihood
pythia1B_crit_base_ll <- get_loglikelihood(pythia1B_non_crit_base_model, rc_rt_prev_crit.pythia1B.data, rc_rt_prev_crit.pythia1B.data$mean_rt)
pythia1B_crit_ll <- get_loglikelihood(pythia1B_non_crit_model, rc_rt_prev_crit.pythia1B.data, rc_rt_prev_crit.pythia1B.data$mean_rt)
pythia1B_crit_delta_ll <- pythia1B_crit_ll - pythia1B_crit_base_ll
pythia1B_crit_delta_ll

### All model summary ----
model <- c("GPT2", "1B", "1B-Instruct", "3B", "3B-Instruct")
perplexity <- c(gpt2_perplexity, llama1B_perplexity, llama1B_instruct_perplexity, llama3B_perplexity, llama3B_instruct_perplexity)

#### all data ----
gam_predictions.data <- bind_rows(
  tibble(
    model = "GPT2",
    surprisal = gpt2_prob_new_data$surprisal,
    mean_rt = gpt2_predictions$fit,
    se = gpt2_predictions$se.fit
  ),
  tibble(
    model = "1B",
    surprisal = llama1B_prob_new_data$surprisal,
    mean_rt = llama1B_predictions$fit,
    se = llama1B_predictions$se.fit
  ),
  tibble(
    model = "3B",
    surprisal = llama3B_prob_new_data$surprisal,
    mean_rt = llama3B_predictions$fit,
    se = llama3B_predictions$se.fit
  ),
  tibble(
    model = "1B-Instruct",
    surprisal = llama1B_instruct_prob_new_data$surprisal,
    mean_rt = llama1B_instruct_predictions$fit,
    se = llama1B_instruct_predictions$se.fit
  ),
  tibble(
    model = "3B-Instruct",
    surprisal = llama3B_instruct_prob_new_data$surprisal,
    mean_rt = llama3B_instruct_predictions$fit,
    se = llama3B_instruct_predictions$se.fit
  )
) %>%
  arrange(model, surprisal)

gam_prediction_graph <- ggplot(data=gam_predictions.data %>% 
         filter(model!="pythia1b") %>% 
         arrange(model, surprisal),
       aes(x=surprisal, y=mean_rt, color=model, fill=model,group=model))+
  geom_line(size=1)  +
  geom_ribbon(
              aes(ymin=mean_rt-1.96*se,
                  ymax=mean_rt+1.96*se),
              alpha=0.3)+ 
  facet_grid(.~model,scales = "free") +
  labs(x="Surprisal",
       y="Reading time") +
  scale_color_manual(values=cbPalette, guide="none") +
  scale_fill_manual(values=cbPalette, guide="none")
gam_prediction_graph
ggsave(gam_prediction_graph, file="../graphs/gam_prediction_graph.pdf", width=8, height=3)

#### delta ll ----
gam_delta_ll <- c(gpt2_gam_delta_ll, llama1B_gam_delta_ll, llama1B_instruct_gam_delta_ll, llama3B_gam_delta_ll, llama3B_instruct_gam_delta_ll)
gam_delta_ll.data <- data.frame(model=model, delta_ll = gam_delta_ll, perplexity = perplexity) %>% 
  arrange(model, delta_ll)

gam_delta_ll_graph <- ggplot(gam_delta_ll.data,
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
ggsave(gam_delta_ll_graph, file="../graphs/gam_delta_ll_graph.pdf", width=8, height=4)
cor(gam_delta_ll, perplexity)

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
gam_delta_ll_cv <- c(delta_ll_gpt2_gam_cv, delta_ll_llama1B_gam_cv, delta_ll_llama1B_instruct_gam_cv, delta_ll_llama3B_gam_cv, delta_ll_llama3B_instruct_gam_cv,delta_ll_pythia1B_gam_cv)
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

gam_delta_ll_cv_graph <- ggplot(gam_delta_ll_cv_summary %>% 
                              filter(model != "pythia1b"),
                            aes(x=perplexity,
                                y=Mean)) +
  geom_smooth(method="lm", formula=y~x,se=TRUE, color="black") +
  geom_point(aes(color=model),size=5)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  labs(x="Perplexity",
       y="ΔLogLik") +
  scale_color_manual(values=cbPalette) +
  theme(legend.text = element_text(size=10))
gam_delta_ll_cv_graph 
ggsave(gam_delta_ll_cv_graph, file="../graphs/gam_delta_ll_cv_graph.pdf", width=8, height=4)



#### critical regions ---- 
critical_loglike <- c(gpt2_crit_delta_ll, llama1B_crit_delta_ll, llama3B_crit_delta_ll, llama1B_instruct_crit_delta_ll, llama3B_instruct_crit_delta_ll, pythia1B_crit_delta_ll)
critical_delta_ll <- data.frame(model = model, delta_ll = critical_loglike)

delta_ll_graph <- ggplot(elta_ll,
                         aes(x=model,
                             y=loglike)) + 
  geom_point()+
  labs(x="Model",
       y="ΔLogLik")
delta_ll_graph  
ggsave(delta_ll_graph, file="../graphs/delta_ll_graph.pdf", width=8, height=4)


## linear models ----
# three previous words

# baseline_lm = mean_rt ~ wordlen + prev_wordlen + prev2_wordlen + log_freq + prev_freq + prev2_freq
# full_lm = mean_rt ~ surprisal + prev_surp + prev2_surp + wordlen + prev_wordlen + prev2_wordlen + log_freq + prev_freq + prev2_freq

# using unigram surprisals
baseline_lm = mean_rt ~ wordlen + prev1_wordlen + prev2_wordlen + surp_sum + prev1_surpsum + prev2_surpsum
full_lm = mean_rt ~ surprisal + prev1_surp + prev2_surp + wordlen + prev1_wordlen + prev2_wordlen + surp_sum + prev1_surpsum + prev2_surpsum

### GPT2 ----
gpt2_rt_linear_graph <- ggplot(rc_rt.gpt2.data,
                               aes(x=surprisal,
                                   y=mean_rt)) +
  geom_point(size=1,alpha=0.6)+
  geom_smooth(color="black", method="lm", se=TRUE) +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
gpt2_rt_linear_graph

rc_rt.gpt2.data <- rc_rt.gpt2.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO", 
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                  "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.gpt2.data <- get_prev(rc_rt.gpt2.data,2)

ggplot(data=data.frame(actual_rt=rc_rt_prev.gpt2.data$mean_rt,
                       surprisal=rc_rt_prev.gpt2.data$surprisal),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1, alpha=0.6)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(color="black", method="lm", formula=y~x,se=TRUE,size=1)+
  labs(x="Surprisal",
       y="Reading time")

lm_base <- lm(baseline_lm, data=rc_rt_prev.gpt2.data)
gpt2_lm_full <- lm(full_lm, data=rc_rt_prev.gpt2.data)
AIC(lm_base)
AIC(gpt2_lm_full)
anova(lm_base, gpt2_lm_full,test="LRT")

## model comparison 
gpt2_lm_ll <- logLik(gpt2_lm_full)
lm_base_ll <- logLik(lm_base)
gpt2_lm_delta_ll <-gpt2_lm_ll - lm_base_ll
gpt2_lm_delta_ll

# plot the predictions
gpt2_predictions <- predict(gpt2_lm_full, newdata=rc_rt_prev.gpt2.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(surprisal=rc_rt_prev.gpt2.data$surprisal,
                       mean_rt=gpt2_predictions$fit),
       aes(x=surprisal,
           y=mean_rt))+
  geom_point(size=1,alpha=.6) +
  geom_smooth(method="lm",formula=y~x,se=TRUE,size=1)+
  # geom_smooth(method="gam",formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="Surprisal",
       y="predicted reading time")

## cross-validation
gpt2_lm_cv <- cv_gam(full_lm, baseline_lm, rc_rt_prev.gpt2.data)
rmse_gpt2_lm_cv <- gpt2_lm_cv$rmse_vec
delta_ll_gpt2_lm_cv <- gpt2_lm_cv$delta_ll

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.gpt2.data <- rc_rt.gpt2.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.gpt2.data <- get_prev(rc_rt.gpt2.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit linear model on words in non-critical regions
gpt2_non_crit_lm <- lm(full_lm, data=rc_rt_prev_non_crit.gpt2.data)
summary(gpt2_non_crit_lm)

# predict the surprisal of words in critical regions using the full lm model (non critical)
gpt2_crit_predictions <- predict(gpt2_non_crit_lm, newdata=rc_rt_prev_crit.gpt2.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.gpt2.data$mean_rt,
                       predicted_rt=gpt2_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="lm", formula=y~x,se=TRUE,size=1,color="black")+
  labs(x="Actual reading time",
       y="Predicted reading time")

get_loglikelihood(gpt2_non_crit_lm, rc_rt_prev_crit.gpt2.data , rc_rt_prev_crit.gpt2.data$mean_rt)

rmse_gpt2 <- sqrt(mean((gpt2_crit_predictions$fit - rc_rt_prev_crit.gpt2.data$mean_rt)^2))
rmse_gpt2
mae_gpt2 <- mean(abs(gpt2_crit_predictions$fit - rc_rt_prev_crit.gpt2.data$mean_rt))
SST_gpt2 <- sum((rc_rt_prev_crit.gpt2.data$mean_rt - mean(rc_rt_prev_crit.gpt2.data$mean_rt))^2)
SSE_gpt2 <- sum((gpt2_crit_predictions$fit - rc_rt_prev_crit.gpt2.data$mean_rt)^2)
r_squared_test_gpt2 <- 1 - (SSE_gpt2 / SST_gpt2)

### Llama-3.2-1B ----
rc_rt.llama1B.data <- rc_rt.llama1B.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3",
                                  "RC_VERB+4", "RC_VERB+5")))

llama1B_rt_linear_graph <- ggplot(rc_rt.llama1B.data,
                                  aes(x=surprisal, 
                                      y=mean_rt)) +
  geom_smooth(color="black", method="lm") +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
llama1B_rt_linear_graph

#### predicting using all words ----
rc_rt_prev.llama1B.data <- get_prev(rc_rt.llama1B.data,2)

ggplot(data=data.frame(actual_rt=rc_rt_prev.llama1B.data$mean_rt,
                       surprisal=rc_rt_prev.llama1B.data$surprisal),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="lm", formula=y~x,se=TRUE,size=1)+
  labs(x="Surprisal",
       y="Reading time")

llama1B_lm_full <- lm(full_lm, data=rc_rt_prev.llama1B.data)
AIC(lm_base)
AIC(llama1B_lm_full) 
anova(lm_base, llama1B_lm_full,test="LRT")

## model comparison 
llama1B_lm_ll <- logLik(llama1B_lm_full)
llama1B_lm_delta_ll <- llama1B_lm_ll - lm_base_ll
llama1B_lm_delta_ll

# plot the predictions
llama1B_predictions <- predict(llama1B_lm_full, newdata=rc_rt_prev.llama1B.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(surprisal=rc_rt_prev.llama1B.data$surprisal,
                       mean_rt=llama1B_predictions$fit),
       aes(x=surprisal,
           y=mean_rt))+
  geom_point(size=1,alpha=.6) +
  geom_smooth(method="lm",formula=y~x,se=TRUE,size=1)+
  # geom_smooth(method="gam",formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="Surprisal",
       y="predicted reading time")

## cross-validation
llama1B_lm_cv <- cv_gam(full_lm, baseline_lm, rc_rt_prev.llama1B.data)
rmse_llama1B_lm_cv <- llama1B_lm_cv$rmse_vec
delta_ll_llama1B_lm_cv <- llama1B_lm_cv$delta_ll

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.llama1B.data <- rc_rt.llama1B.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.llama1B.data <- get_prev(rc_rt.llama1B.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit linear model on words in non-critical regions
llama1B_non_crit_lm <- lm(full_lm, data=rc_rt_prev_non_crit.llama1B.data)
summary(llama1B_non_crit_lm)

# predict the surprisal of words in critical regions using the full lm model 
llama1B_crit_predictions <- predict(llama1B_non_crit_lm, newdata=rc_rt_prev_crit.llama1B.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.llama1B.data$mean_rt,
                       predicted_rt=llama1B_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="lm", formula=y~x,se=TRUE,size=1)+
  labs(x="Actual reading time",
       y="Predicted reading time")

get_loglikelihood(llama1B_non_crit_lm, rc_rt_prev_crit.llama1B.data , rc_rt_prev_crit.llama1B.data$mean_rt)

rmse_llama1B <- sqrt(mean((llama1B_crit_predictions$fit - rc_rt_prev_crit.llama1B.data$mean_rt)^2))
mae_llama1B <- mean(abs(llama1B_crit_predictions$fit - rc_rt_prev_crit.llama1B.data$mean_rt))
SST_llama1B <- sum((rc_rt_prev_crit.llama1B.data$mean_rt - mean(rc_rt_prev_crit.llama1B.data$mean_rt))^2)
SSE_llama1B <- sum((llama1B_crit_predictions$fit - rc_rt_prev_crit.llama1B.data$mean_rt)^2)
r_squared_test_llama1B <- 1 - (SSE_llama1B / SST_llama1B)


### Llama-3.2-3B ----
rc_rt.llama3B.data <- rc_rt.llama3B.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                  "RC_VERB+4", "RC_VERB+5")))

llama3B_rt_linear_graph <- ggplot(rc_rt.llama3B.data,
                                  aes(x=surprisal, 
                                      y=mean_rt)) +
  geom_smooth(color="black", method="lm") +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
llama3B_rt_linear_graph

#### predicting using all words ----
rc_rt_prev.llama3B.data <- get_prev(rc_rt.llama3B.data, 2)

ggplot(data=data.frame(actual_rt=rc_rt_prev.llama3B.data$mean_rt,
                       surprisal=rc_rt_prev.llama3B.data$surprisal),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="lm", formula=y~x,se=TRUE,size=1)+
  labs(x="Surprisal",
       y="Reading time")

llama3B_lm_full <- lm(full_lm, data=rc_rt_prev.llama3B.data)
AIC(lm_base) 
AIC(llama3B_lm_full)
anova(lm_base, llama3B_lm_full,test="LRT")

llama3B_lm_ll <- logLik(llama3B_lm_full)
llama3B_lm_delta_ll <- llama3B_lm_ll - lm_base_ll
llama3B_lm_delta_ll

# plot the predictions
llama3B_predictions <- predict(llama3B_lm_full, newdata=rc_rt_prev.llama3B.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(surprisal=rc_rt_prev.llama3B.data$surprisal,
                       mean_rt=llama3B_predictions$fit),
       aes(x=surprisal,
           y=mean_rt))+
  geom_point(size=1,alpha=.6) +
  geom_smooth(method="lm",formula=y~x,se=TRUE,size=1)+
  # geom_smooth(method="gam",formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="Surprisal",
       y="predicted reading time")

## cross-validation
llama3B_lm_cv <- cv_gam(full_lm, baseline_lm, rc_rt_prev.llama3B.data)
rmse_llama3B_lm_cv <- llama3B_lm_cv$rmse_vec
delta_ll_llama3B_lm_cv <- llama3B_lm_cv$delta_ll

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.llama3B.data <- rc_rt.llama3B.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.llama3B.data <- get_prev(rc_rt.llama3B.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit linear model on words in non-critical regions
## full model
llama3B_non_crit_lm <- lm(full_lm, data=rc_rt_prev_non_crit.llama3B.data)
summary(llama3B_non_crit_lm)

# predict the surprisal of words in critical regions using the full lm model 
llama3B_crit_predictions <- predict(llama3B_non_crit_lm, newdata=rc_rt_prev_crit.llama3B.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.llama3B.data$mean_rt,
                       predicted_rt=llama3B_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="lm", formula=y~x,se=TRUE,size=1)+
  labs(x="Actual reading time",
       y="Predicted reading time")

get_loglikelihood(llama3B_non_crit_lm, rc_rt_prev_crit.llama3B.data , rc_rt_prev_crit.llama3B.data$mean_rt)

rmse_llama3B <- sqrt(mean((llama3B_crit_predictions$fit - rc_rt_prev_crit.llama3B.data$mean_rt)^2))
mae_llama3B <- mean(abs(llama3B_crit_predictions$fit - rc_rt_prev_crit.llama3B.data$mean_rt))
SST_llama3B <- sum((rc_rt_prev_crit.llama3B.data$mean_rt - mean(rc_rt_prev_crit.llama3B.data$mean_rt))^2)
SSE_llama3B <- sum((llama3B_crit_predictions$fit - rc_rt_prev_crit.llama3B.data$mean_rt)^2)
r_squared_test_llama3B <- 1 - (SSE_llama3B / SST_llama3B)

### Llama-3.2-1B-Instruct ----
rc_rt.llama1B_instruct.data <- rc_rt.llama1B_instruct.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", 
                                  "RC_VERB+4", "RC_VERB+5")))

llama1B_instruct_rt_linear_graph <- ggplot(rc_rt.llama1B_instruct.data,
                                           aes(x=surprisal,
                                               y=mean_rt)) +
  geom_smooth(color="black", method="lm") +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
llama1B_instruct_rt_linear_graph


#### predicting using all words ----
rc_rt_prev.llama1B_instruct.data <- get_prev(rc_rt.llama1B_instruct.data, 2)

ggplot(data=data.frame(actual_rt=rc_rt_prev.llama1B_instruct.data$mean_rt,
                       surprisal=rc_rt_prev.llama1B_instruct.data$surprisal),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="lm", formula=y~x,se=TRUE,size=1)+
  labs(x="Surprisal",
       y="Reading time")

llama1B_instruct_lm_full <- lm(full_lm, data=rc_rt_prev.llama1B_instruct.data)
AIC(lm_base)
AIC(llama1B_instruct_lm_full)
anova(lm_base, llama1B_instruct_lm_full,test="LRT")

llama1B_instruct_lm_ll <- logLik(llama1B_instruct_lm_full)
llama1B_instruct_lm_delta_ll <- llama1B_instruct_lm_ll - lm_base_ll
llama1B_instruct_lm_delta_ll

# plot the predictions
llama1B_instruct_predictions <- predict(llama1B_instruct_lm_full, newdata=rc_rt_prev.llama1B_instruct.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(surprisal=rc_rt_prev.llama1B_instruct.data$surprisal,
                       mean_rt=llama1B_instruct_predictions$fit),
       aes(x=surprisal,
           y=mean_rt))+
  geom_point(size=1,alpha=.6) +
  geom_smooth(method="lm",formula=y~x,se=TRUE,size=1)+
  # geom_smooth(method="gam",formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="Surprisal",
       y="predicted reading time")

## cross-validation
llama1B_instruct_lm_cv <- cv_gam(full_lm, baseline_lm, rc_rt_prev.llama1B_instruct.data)
rmse_llama1B_instruct_lm_cv <- llama1B_instruct_lm_cv$rmse_vec
delta_ll_llama1B_instruct_lm_cv <- llama1B_instruct_lm_cv$delta_ll


#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.llama1B_instruct.data <- rc_rt.llama1B_instruct.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.llama1B_instruct.data <- get_prev(rc_rt.llama1B_instruct.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit linear model on words in non-critical regions
## full model
llama1B_instruct_non_crit_lm <- lm(full_lm, data=rc_rt_prev_non_crit.llama1B_instruct.data)
summary(llama1B_instruct_non_crit_lm)

# predict the surprisal of words in critical regions using the full lm model 
llama1B_instruct_crit_predictions <- predict(llama1B_instruct_non_crit_lm, newdata=rc_rt_prev_crit.llama1B_instruct.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.llama1B_instruct.data$mean_rt,
                       predicted_rt=llama1B_instruct_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="lm", formula=y~x,se=TRUE,size=1)+
  labs(x="Actual reading time",
       y="Predicted reading time")

get_loglikelihood(llama1B_instruct_non_crit_lm, rc_rt_prev_crit.llama1B_instruct.data , rc_rt_prev_crit.llama1B_instruct.data$mean_rt)

rmse_llama1B_instruct <- sqrt(mean((llama1B_instruct_crit_predictions$fit - rc_rt_prev_crit.llama1B_instruct.data$mean_rt)^2))
mae_llama1B_instruct <- mean(abs(llama1B_instruct_crit_predictions$fit - rc_rt_prev_crit.llama1B_instruct.data$mean_rt))
SST_llama1B_instruct <- sum((rc_rt_prev_crit.llama1B_instruct.data$mean_rt - mean(rc_rt_prev_crit.llama1B_instruct.data$mean_rt))^2)
SSE_llama1B_instruct <- sum((llama1B_instruct_crit_predictions$fit - rc_rt_prev_crit.llama1B_instruct.data$mean_rt)^2)
r_squared_test_llama1B_instruct <- 1 - (SSE_llama1B_instruct / SST_llama1B_instruct)

### Llama-3.2-3B-Instruct ----
rc_rt.llama3B_instruct.data <- rc_rt.llama3B_instruct.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",
                                  "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3",
                                  "RC_VERB+4", "RC_VERB+5")))

llama3B_instruct_rt_linear_graph <- ggplot(rc_rt.llama3B_instruct.data,
                                           aes(x=surprisal,
                                               y=mean_rt)) +
  geom_point(size=1, alpha=0.6)+
  geom_smooth(color="black", method="lm") +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
llama3B_instruct_rt_linear_graph

#### predicting using all words ----
rc_rt_prev.llama3B_instruct.data <- get_prev(rc_rt.llama3B_instruct.data, 2)

ggplot(data=data.frame(actual_rt=rc_rt_prev.llama3B_instruct.data$mean_rt,
                       surprisal=rc_rt_prev.llama3B_instruct.data$surprisal),
       aes(x=surprisal,
           y=actual_rt))+
  geom_point(size=1)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="lm", formula=y~x,se=TRUE,size=1,color="black")+
  labs(x="Surprisal",
       y="Reading time")

llama3B_instruct_lm_full <- lm(full_lm, data=rc_rt_prev.llama3B_instruct.data)
AIC(lm_base)
AIC(llama3B_instruct_lm_full)
anova(lm_base, llama3B_instruct_lm_full,test="LRT")

llama3B_instruct_lm_ll <- logLik(llama3B_instruct_lm_full)
llama3B_instruct_lm_delta_ll <- llama3B_instruct_lm_ll - lm_base_ll
llama3B_instruct_lm_delta_ll

# plot the predictions
llama3B_instruct_predictions <- predict(llama3B_instruct_lm_full, newdata=rc_rt_prev.llama3B_instruct.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(surprisal=rc_rt_prev.llama3B_instruct.data$surprisal,
                       mean_rt=llama3B_instruct_predictions$fit),
       aes(x=surprisal,
           y=mean_rt))+
  geom_point(size=1,alpha=.6) +
  geom_smooth(method="lm",formula=y~x,se=TRUE,size=1,color="black")+
  # geom_smooth(method="gam",formula=y~s(x,bs="cs"),se=TRUE,size=1)+
  labs(x="Surprisal",
       y="predicted reading time")

llama3B_instruct_lm_cv <- cv_gam(full_lm, baseline_lm, rc_rt_prev.llama3B_instruct.data)
rmse_llama3B_instruct_lm_cv <- llama3B_instruct_lm_cv$rmse_vec
delta_ll_llama3B_instruct_lm_cv <- llama3B_instruct_lm_cv$delta_ll

#### predicting using words outside the critical region ----
# separate into two dataframes with non-critical (training) and critical (testing) regions
rc_rt_prev_non_crit.llama3B_instruct.data <- rc_rt.llama3B_instruct.data %>% 
  filter(crit != "RC_VERB") %>% 
  get_prev(2) %>% 
  filter(!crit %in% c("RC_VERB+1", "RC_VERB+2"))

rc_rt_prev_crit.llama3B_instruct.data <- get_prev(rc_rt.llama3B_instruct.data,2) %>% 
  filter(crit %in% c("RC_VERB","RC_VERB+1", "RC_VERB+2"))

# fit linear model on words in non-critical regions
## full model
llama3B_instruct_non_crit_lm <- lm(full_lm, data=rc_rt_prev_non_crit.llama3B_instruct.data)
summary(llama3B_instruct_non_crit_lm)

# predict the surprisal of words in critical regions using the full lm model 
llama3B_instruct_crit_predictions <- predict(llama3B_instruct_non_crit_lm, newdata=rc_rt_prev_crit.llama3B_instruct.data,type="response", se.fit=TRUE)

ggplot(data=data.frame(actual_rt=rc_rt_prev_crit.llama3B_instruct.data$mean_rt,
                       predicted_rt=llama3B_instruct_crit_predictions$fit),
       aes(x=actual_rt,
           y=predicted_rt))+
  geom_point(size=1,alpha=.6)+
  # scale_x_continuous(limits = c(300,750))+
  # scale_y_continuous(limits = c(300,750))+
  geom_smooth(method="lm", formula=y~x,se=TRUE,size=1,color="black")+
  labs(x="Actual reading time",
       y="Predicted reading time")

get_loglikelihood(llama3B_instruct_non_crit_lm, rc_rt_prev_crit.llama3B_instruct.data , rc_rt_prev_crit.llama3B_instruct.data$mean_rt)

rmse_llama3B_instruct <- sqrt(mean((llama3B_instruct_crit_predictions$fit - rc_rt_prev_crit.llama3B_instruct.data$mean_rt)^2))
mae_llama3B_instruct <- mean(abs(llama3B_instruct_crit_predictions$fit - rc_rt_prev_crit.llama3B_instruct.data$mean_rt))
SST_llama3B_instruct <- sum((rc_rt_prev_crit.llama3B_instruct.data$mean_rt - mean(rc_rt_prev_crit.llama3B_instruct.data$mean_rt))^2)
SSE_llama3B_instruct <- sum((llama3B_instruct_crit_predictions$fit - rc_rt_prev_crit.llama3B_instruct.data$mean_rt)^2)
r_squared_test_llama3B_instruct <- 1 - (SSE_llama3B_instruct / SST_llama3B_instruct)

### All model summary ----
model <- c("gpt2", "llama1b", "llama3b", "llama1b-instruct", "llama3b-instruct")
perplexity <- c(gpt2_perplexity, llama1B_perplexity, llama1B_instruct_perplexity, llama3B_perplexity, llama3B_instruct_perplexity)

#### all data ----
lm_delta_ll <- c(gpt2_lm_delta_ll, llama1B_lm_delta_ll, llama1B_instruct_lm_delta_ll, llama3B_lm_delta_ll, llama3B_instruct_lm_delta_ll)
lm_delta_ll.data <- data.frame(model=model, delta_ll = lm_delta_ll, perplexity = perplexity)

lm_delta_ll_graph <- ggplot(lm_delta_ll.data,
                             aes(x=perplexity,
                                 y=delta_ll)) + 
  geom_point(aes(color=model),size=5)+
  geom_smooth(method="lm", formula=y~x,se=TRUE, color="black") +
  labs(x="Perplexity",
       y="ΔLogLik") +
  scale_color_manual(values=cbPalette)
lm_delta_ll_graph  
ggsave(lm_delta_ll_graph, file="../graphs/lm_delta_ll_graph.pdf", width=8, height=4)

#### cross-validation ----
lm_delta_ll_cv <- c(delta_ll_gpt2_lm_cv, delta_ll_llama1B_lm_cv, delta_ll_llama1B_instruct_lm_cv, delta_ll_llama3B_lm_cv, delta_ll_llama1B_instruct_lm_cv)
lm_delta_ll_cv.data <- data.frame(model=model, delta_ll = lm_delta_ll_cv, perplexity = perplexity) %>% 
  arrange(model, delta_ll)
lm_delta_ll_cv_summary <- lm_delta_ll_cv.data %>% 
  group_by(model, perplexity) %>% 
  summarize(Mean = mean(delta_ll),
            CILow = ci.low(delta_ll),
            CIHigh = ci.high(delta_ll)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

lm_delta_ll_cv_graph <- ggplot(lm_delta_ll_cv_summary,
                            aes(x=perplexity,
                                y=Mean)) +
  geom_smooth(method="lm", formula=y~x,se=TRUE, color="black") +
  geom_point(aes(color=model),size=5)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  labs(x="Perplexity",
       y="ΔLogLik") +
  scale_color_manual(values=cbPalette)
lm_delta_ll_cv_graph 
ggsave(lm_delta_ll_cv_graph, file="../graphs/lm_delta_ll_cv_graph.pdf", width=8, height=4)
