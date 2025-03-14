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
library(mgcv)
library(brms)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
## GPT-2 ----
rc_rt.gpt2.data <- read.csv("../../../data/rc_rt/rc_rt_combined_gpt2_3.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.gpt2.summary <- rc_rt.gpt2.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",                                        "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))

## Llama3.2-1B ----
rc_rt.llama1B.data <- read.csv("../../../data/rc_rt/rc_rt_combined_Llama-3.2-1B_1.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.llama1B.summary <- rc_rt.llama1B.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit = fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",                                        "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))

## Llama3.2-3B ----
rc_rt.llama3B.data <- read.csv("../../../data/rc_rt/rc_rt_combined_Llama-3.2-3B_1.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.llama3B.summary <- rc_rt.llama3B.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit = fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",                                        "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))


## Llama3.2-1B-Instruct ----
rc_rt.llama1B_instruct.data <- read.csv("../../../data/rc_rt/rc_rt_combined_Llama-3.2-1B-Instruct_1.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.llama1B_instruct.summary <- rc_rt.llama1B_instruct.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit = fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",                                        "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))

## Llama3.2-3B-Instruct ----
rc_rt.llama3B_instruct.data <- read.csv("../../../data/rc_rt/rc_rt_combined_Llama-3.2-3B-Instruct_1.csv", header=TRUE) %>% 
  na.omit() %>% 
  select(-c(logprob, item_matrix, item_embed, sentence, logprob, word_prob)) %>% 
  mutate(surprisal = -prob,
         crit = if_else(crit == "NONE", "RC_VERB+5", crit)) %>% 
  mutate(crit = factor(crit))

rc_rt.llama3B_instruct.summary <- rc_rt.llama3B_instruct.data %>% 
  # filter(!crit %in% c("NONE", "SUBJ", "RC_VERB+3", "RC_VERB+4")) %>% 
  group_by(crit, cond) %>% 
  summarize(Mean = mean(surprisal),
            CILow = ci.low(surprisal),
            CIHigh = ci.high(surprisal)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh) %>% 
  mutate(crit = fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO", "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))

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


# 3. Analysis ----
## GAM models ----
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
    mutate(prev_surp = lag(surprisal, order_by = crit),
           prev_word = lag(word, order_by=crit),
           prev_wordlen = lag(wordlen, order_by=crit),
           # prev_freq = lag(log_freq, order_by = crit),
           
           prev2_word = lag(prev_word, order_by = crit),
           prev2_wordlen = lag(prev_wordlen, order_by = crit),
           prev2_surp = lag(prev_surp, order_by = crit)) #,
           # prev2_freq = lag(prev_freq, order_by = crit))
  if(num_prev==3) { # if getting the measures of the previous three words 
    prev_df <- prev_df %>% 
      mutate(prev3_word = lag(prev2_word, order_by = crit),
             prev3_wordlen = lag(prev2_wordlen, order_by = crit),
             prev3_surp = lag(prev2_surp, order_by = crit)) %>%  # ,
             # prev3_freq = lag(prev2_freq, order_by = crit))  %>% 
      select(-c(prev_word,prev2_word,prev3_word)) %>%
      drop_na() %>% 
      ungroup()
  } else {
    prev_df <- prev_df %>% 
      select(-c(prev_word,prev2_word)) %>%
      drop_na() %>% 
      ungroup()
  }
  return(prev_df)
}

get_new_df <- function(dataframe, length_out, num_prev) {
  new_df <- data.frame(surprisal = seq(min(dataframe$surprisal),
                                       max(dataframe$surprisal),
                                       length.out=length_out),
                       prev_surp = seq(min(dataframe$prev_surp),
                                       max(dataframe$prev_surp),
                                       length.out=length_out),
                       prev2_surp = seq(min(dataframe$prev2_surp),
                                        max(dataframe$prev2_surp),
                                        length.out = length_out),
                       wordlen = seq(min(dataframe$wordlen), 
                                     max(dataframe$wordlen), 
                                     length.out = length_out), 
                       prev_wordlen = seq(min(dataframe$prev_wordlen),
                                          max(dataframe$prev_wordlen),
                                          length.out = length_out), 
                       prev2_wordlen = seq(min(dataframe$prev2_wordlen),
                                           max(dataframe$prev2_wordlen),
                                           length.out = length_out)) #,
                       # log_freq = seq(min(dataframe$log_freq),
                       #            max(dataframe$log_freq),
                       #            length.out = length_out),
                       # prev_freq = seq(min(dataframe$prev_freq),
                       #                 max(dataframe$prev_freq),
                       #                 length.out = length_out),
                       # prev2_freq = seq(min(dataframe$prev2_freq),
                       #                  max(dataframe$prev2_freq),
                       #                  length.out = length_out))
  if(num_prev==3) {
    new_df <- new_df %>% 
      mutate(prev3_surp = seq(min(dataframe$prev3_surp),
                              max(dataframe$prev3_surp),
                              length.out = length_out), 
             prev3_wordlen = seq(min(dataframe$prev3_wordlen),
                                 max(dataframe$prev3_wordlen),
                                 length.out=length_out)) # ,
             # prev3_freq = seq(min(dataframe$prev3_freq),
             #                  max(dataframe$prev3_freq),
             #                  length.out = length_out))
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


### GPT2 ----
rc_rt.gpt2.data <- rc_rt.gpt2.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",                                        "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.gpt2.data <- get_prev(rc_rt.gpt2.data,3)

gpt2_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) + 
                    s(prev_surp, bs="cr", k=10) + 
                    s(prev2_surp, bs="cr", k=10) +
                    s(prev3_surp, bs="cr", k=10) +
                    te(wordlen, bs="cr") + 
                    te(prev_wordlen, bs="cr") + 
                    te(prev2_wordlen, bs="cr") + 
                    te(prev3_wordlen, bs="cr"), # +
                    # te(log_freq, bs="cr") +
                    # te(prev_freq, bs="cr") +
                    # te(prev2_freq, bs="cr") +
                    # te(prev3_freq, bs="cr"),
                  data=rc_rt_prev.gpt2.data)
summary(gpt2_model)
gam.check(gpt2_model)

# for visualization purpose
## create the dataframe for prediction and ploting
gpt2_prob_new_data <- get_new_df(rc_rt_prev.gpt2.data,100,3)
## predict using the new dataframe
gpt2_predictions <- predict(gpt2_model, newdata=gpt2_prob_new_data,type="response",
                       se.fit=TRUE)
## plot the predictions
gpt2_rt_graph <- plot_predictions(gpt2_prob_new_data, gpt2_predictions)
gpt2_rt_graph
ggsave(gpt2_rt_graph, file="../graphs/gpt2_rt_graph.pdf", width=8, height=4)

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
gpt2_non_crit_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) + 
                             s(prev_surp, bs="cr", k=10) +
                             s(prev2_surp, bs="cr", k=10) +
                             te(wordlen, bs="cr") + 
                             te(prev_wordlen, bs="cr") + 
                             te(prev2_wordlen, bs="cr"), # +
                             # te(log_freq, bs="cr") + 
                             # te(prev_freq, bs="cr") + 
                             # te(prev2_freq, bs="cr"),
                  data=rc_rt_prev_non_crit.gpt2.data)
summary(gpt2_non_crit_model)
gam.check(gpt2_non_crit_model)

## base model
gpt2_non_crit_base_model <- gam(mean_rt ~ te(wordlen, bs="cr") + 
                                  te(prev_wordlen, bs = "cr") +
                                  te(prev2_wordlen, bs = "cr"), # +
                                  # te(log_freq, bs = "cr") +
                                  # te(prev_freq, bs = "cr") +
                                  # te(prev2_freq, bs = "cr"), 
                           data=rc_rt_prev_non_crit.gpt2.data)
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
## plot predictions
plot_predictions(gpt2_non_crit_new_data, gpt2_non_crit_predictions)

## predict the surprisal of words in non-critical regions using the base gam model 
gpt2_non_crit_base_predictions <- predict(gpt2_non_crit_base_model, newdata=gpt2_non_crit_new_data,type="response",
                                     se.fit=TRUE)
## plot base model 
plot_predictions(gpt2_non_crit_new_data, gpt2_non_crit_base_predictions)

# for visualization purpose -- critical regions
## create dataframe with equal entries across columns
gpt2_crit_new_data <- get_new_df(rc_rt_prev_crit.gpt2.data, 100, 2)

## predict the surprisal of words in critical regions using the full gam model 
gpt2_crit_predictions <- predict(gpt2_non_crit_model, newdata=gpt2_crit_new_data,type="response",
                                 se.fit=TRUE)
## plot full model
plot_predictions(gpt2_crit_new_data, gpt2_crit_predictions)

## predict the surprisal of words in critical regions using the full gam model 
gpt2_crit_base_predictions <- predict(gpt2_non_crit_base_model, newdata=gpt2_crit_new_data,type="response",
                                 se.fit=TRUE)
## plot base model
plot_predictions(gpt2_crit_new_data, gpt2_crit_base_predictions)

# compute the delta log-likelihood
gpt2_crit_base_ll <- get_loglikelihood(gpt2_non_crit_base_model, rc_rt_prev_crit.gpt2.data, rc_rt_prev_crit.gpt2.data$mean_rt)
gpt2_crit_ll <- get_loglikelihood(gpt2_non_crit_model, rc_rt_prev_crit.gpt2.data, rc_rt_prev_crit.gpt2.data$mean_rt)
gpt2_crit_delta_ll <- gpt2_crit_ll - gpt2_crit_base_ll
gpt2_crit_delta_ll

### Llama3.2-1B ----
rc_rt.llama1B.data <- rc_rt.llama1B.data %>% 
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",                                        "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.llama1B.data <- get_prev(rc_rt.llama1B.data,3)

llama1B_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) + 
                       s(prev_surp, bs="cr", k=10) + 
                       s(prev2_surp, bs="cr", k=10) + 
                       s(prev3_surp, bs="cr", k=10) +
                       te(wordlen, bs="cr") + 
                       te(prev_wordlen, bs="cr") + 
                       te(prev2_wordlen, bs="cr") + 
                       te(prev3_wordlen, bs="cr"), # +
                       # te(log_freq, bs="cr") +
                       # te(prev_freq, bs="cr") +
                       # te(prev2_freq, bs="cr") +
                       # te(prev3_freq, bs="cr"),
                  data=rc_rt_prev.llama1B.data)
summary(llama1B_model)
gam.check(llama1B_model)

# for visualization purpose
## create the dataframe for prediction and ploting
llama1B_prob_new_data <- get_new_df(rc_rt_prev.llama1B.data, 100, 3)
## predict using the new dataframe
llama1B_predictions <- predict(llama1B_model, newdata=llama1B_prob_new_data,type="response",
                            se.fit=TRUE)
## plot predictions
llama1B_rt_graph <- plot_predictions(llama1B_prob_new_data, llama1B_predictions) 
llama1B_rt_graph
ggsave(llama1B_rt_graph, file="../graphs/llama1B_rt_freq_graph.pdf", width=8, height=4)

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
llama1B_non_crit_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) + 
                                s(prev_surp, bs="cr", k=10) +
                                s(prev2_surp, bs = "cr", k = 10) +
                                te(wordlen, bs = "cr") +
                                te(prev_wordlen, bs = "cr") +
                                te(prev2_wordlen, bs = "cr"), # +
                                # te(log_freq, bs = "cr") +
                                # te(prev_freq, bs = "cr") +
                                # te(prev2_freq, bs = "cr"),
                           data=rc_rt_prev_non_crit.llama1B.data)
summary(llama1B_non_crit_model)
gam.check(llama1B_non_crit_model)

## base model
llama1B_non_crit_base_model <- gam(mean_rt ~ te(wordlen, bs="cr") + 
                                     te(prev_wordlen, bs = "cr") +
                                     te(prev2_wordlen, bs = "cr"), # +
                                     # te(log_freq, bs = "cr") +
                                     # te(prev_freq, bs = "cr") +
                                     # te(prev2_freq, bs="cr"),
                                data=rc_rt_prev_non_crit.llama1B.data)
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
## create dataframe with equal entries across columns
llama1B_crit_new_data <- get_new_df(rc_rt_prev_crit.llama1B.data, 100, 2)

## predict the surprisal of words in critical regions using the full gam model 
llama1B_crit_predictions <- predict(llama1B_non_crit_model, newdata=llama1B_crit_new_data,type="response",
                                 se.fit=TRUE)
## plot full model
plot_predictions(llama1B_crit_new_data, llama1B_crit_predictions)

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
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",                                        "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.llama3B.data <- get_prev(rc_rt.llama3B.data, 3)

llama3B_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) + 
                       s(prev_surp, bs="cr", k=10) + 
                       s(prev2_surp, bs="cr", k=10) + 
                       s(prev3_surp, bs="cr", k=10) +
                       te(wordlen, bs="cr") + 
                       te(prev_wordlen, bs="cr") + 
                       te(prev2_wordlen, bs="cr") + 
                       te(prev3_wordlen, bs="cr"), # +
                       # te(log_freq, bs="cr") + 
                       # te(prev_freq, bs="cr") + 
                       # te(prev2_freq, bs="cr") +
                       # te(prev3_freq, bs="cr"),
                     data=rc_rt_prev.llama3B.data)
summary(llama3B_model)
gam.check(llama3B_model)

# for visualization purpose
## create the dataframe for prediction and ploting
llama3B_prob_new_data <- get_new_df(rc_rt_prev.llama3B.data, 100, 3)
## predict using the new dataframe
llama3B_predictions <- predict(llama3B_model, newdata=llama3B_prob_new_data,type="response",
                               se.fit=TRUE)
## plot the predictions
llama3B_rt_graph <- plot_predictions(llama3B_prob_new_data,llama3B_predictions)
llama3B_rt_graph
ggsave(llama3B_rt_graph, file="../graphs/llama3B_rt_graph.pdf", width=8, height=4)

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
llama3B_non_crit_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) + 
                                s(prev_surp, bs="cr", k=10) +
                                s(prev2_surp, bs="cr", k=10) +
                                te(wordlen, bs="cr") + 
                                te(prev_wordlen, bs="cr") + 
                                te(prev2_wordlen, bs="cr"), # +
                                # te(log_freq, bs="cr") + 
                                # te(prev_freq, bs="cr") + 
                                # te(prev2_freq, bs="cr"),
                              data=rc_rt_prev_non_crit.llama3B.data)
summary(llama3B_non_crit_model)
gam.check(llama3B_non_crit_model)

## base model
llama3B_non_crit_base_model <- gam(mean_rt ~ te(wordlen, bs="cr") + 
                                     te(prev_wordlen, bs="cr") + 
                                     te(prev2_wordlen, bs="cr"), # +
                                     # te(log_freq, bs="cr") + 
                                     # te(prev_freq, bs="cr") + 
                                     # te(prev2_freq, bs="cr"),
                                   data=rc_rt_prev_non_crit.llama3B.data)
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
## create dataframe with equal entries across columns
llama3B_crit_new_data <- get_new_df(rc_rt_prev_crit.llama3B.data, 100, 2)

## predict the surprisal of words in critical regions using the full gam model 
llama3B_crit_predictions <- predict(llama3B_non_crit_model, newdata=llama3B_crit_new_data,type="response",
                                    se.fit=TRUE)
## plot full model
plot_predictions(llama3B_crit_new_data, llama3B_crit_predictions)

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
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",                                        "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.llama1B_instruct.data <- get_prev(rc_rt.llama1B_instruct.data, 3)

llama1B_instruct_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) +
                                s(prev_surp, bs="cr", k=10) + 
                                s(prev2_surp, bs = "cr", k = 10) +
                                s(prev3_surp, bs = "cr", k = 10) +
                                te(wordlen, bs = "cr") +
                                te(prev_wordlen, bs = "cr") +
                                te(prev2_wordlen, bs = "cr") +
                                te(prev3_wordlen, bs="cr"), # +
                                # te(log_freq, bs = "cr") +
                                # te(prev_freq, bs = "cr") +
                                # te(prev2_freq, bs = "cr") +
                                # te(prev3_freq, bs = "cr"), 
                              data = rc_rt_prev.llama1B_instruct.data)
summary(llama1B_instruct_model)
gam.check(llama1B_instruct_model)

# for visualization purpose
## create the dataframe for prediction and ploting
llama1B_instruct_prob_new_data <- get_new_df(rc_rt_prev.llama1B_instruct.data, 100, 3)
## predict using the new dataframe
llama1B_instruct_predictions <- predict(llama1B_instruct_model, newdata=llama1B_instruct_prob_new_data,type="response",
                               se.fit=TRUE)
## plot the predictions
llama1B_instruct_rt_graph <- plot_predictions(llama1B_instruct_prob_new_data,llama1B_instruct_predictions)
llama1B_instruct_rt_graph
ggsave(llama1B_instruct_rt_graph, file="../graphs/llama1B_instruct_rt_graph.pdf", width=8, height=4)


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
llama1B_instruct_non_crit_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) + 
                                         s(prev_surp, bs = "cr", k = 10) +
                                         s(prev2_surp, bs = "cr", k = 10) +
                                         te(wordlen, bs = "cr") +
                                         te(prev_wordlen, bs = "cr") +
                                         te(prev2_wordlen, bs = "cr"), # +
                                         # te(log_freq, bs="cr") + 
                                         # te(prev_freq, bs="cr") + 
                                         # te(prev2_freq, bs="cr"), 
                                       data = rc_rt_prev_non_crit.llama1B_instruct.data)
summary(llama1B_instruct_non_crit_model)
gam.check(llama1B_instruct_non_crit_model)

## base model
llama1B_instruct_non_crit_base_model <- gam(mean_rt ~ te(wordlen, bs="cr") + 
                                              te(prev_wordlen, bs="cr") + 
                                              te(prev2_wordlen, bs="cr"), # +
                                              # te(log_freq, bs="cr") + 
                                              # te(prev_freq, bs="cr") + 
                                              # te(prev2_freq, bs="cr"),
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
## create dataframe with equal entries across columns
llama1B_instruct_crit_new_data <- get_new_df(rc_rt_prev_crit.llama1B_instruct.data, 100, 2)

## predict the surprisal of words in critical regions using the full gam model 
llama1B_instruct_crit_predictions <- predict(llama1B_instruct_non_crit_model, 
                                             newdata=llama1B_instruct_crit_new_data,
                                             type="response",
                                             se.fit=TRUE)
## plot full model
plot_predictions(llama1B_instruct_crit_new_data, llama1B_instruct_crit_predictions)

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
  mutate(crit=fct_relevel(crit, c("SUBJ", "MAIN_VERB", "NP1", "OF", "NP2", "WHO",                                        "RC_VERB", "RC_VERB+1", "RC_VERB+2", "RC_VERB+3", "RC_VERB+4", "RC_VERB+5")))

#### predicting using all words ----
rc_rt_prev.llama3B_instruct.data <- get_prev(rc_rt.llama3B_instruct.data, 3)

llama3B_instruct_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) + 
                                s(prev_surp, bs="cr", k=10) + 
                                s(prev2_surp, bs="cr", k=10) + 
                                s(prev3_surp, bs="cr", k=10) +
                                te(wordlen, bs="cr") + 
                                te(prev_wordlen, bs="cr") + 
                                te(prev2_wordlen, bs="cr") + 
                                te(prev3_wordlen, bs="cr"), # +
                                # te(log_freq, bs="cr") + 
                                # te(prev_freq, bs="cr") + 
                                # te(prev2_freq, bs="cr") +
                                # te(prev3_freq, bs="cr"),
                              data=rc_rt_prev.llama3B_instruct.data)
summary(llama3B_instruct_model)
gam.check(llama3B_instruct_model)

# for visualization purpose
## create the dataframe for prediction and ploting
llama3B_instruct_prob_new_data <- get_new_df(rc_rt_prev.llama3B_instruct.data, 100, 3)
## predict using the new dataframe
llama3B_instruct_predictions <- predict(llama3B_instruct_model, newdata=llama3B_instruct_prob_new_data,type="response",
                                        se.fit=TRUE)
## plot the predictions
llama3B_instruct_rt_graph <- plot_predictions(llama3B_instruct_prob_new_data,llama3B_instruct_predictions)
llama3B_instruct_rt_graph
ggsave(llama3B_instruct_rt_graph, file="../graphs/llama3B_instruct_rt_graph.pdf", width=8, height=4)

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
llama3B_instruct_non_crit_model <- gam(mean_rt ~ s(surprisal, bs="cr", k=10) + 
                                         s(prev_surp, bs = "cr", k = 10) +
                                         s(prev2_surp, bs = "cr", k = 10) +
                                         te(wordlen, bs = "cr") +
                                         te(prev_wordlen, bs = "cr") +
                                         te(prev2_wordlen, bs = "cr"), # +
                                         # te(log_freq, bs = "cr") +
                                         # te(prev_freq, bs = "cr") +
                                         # te(prev2_freq, bs = "cr"), 
                                       data = rc_rt_prev_non_crit.llama3B_instruct.data)
summary(llama3B_instruct_non_crit_model)
gam.check(llama3B_instruct_non_crit_model)

## base model
llama3B_instruct_non_crit_base_model <- gam(mean_rt ~ te(wordlen, bs="cr") + 
                                              te(prev_wordlen, bs="cr") + 
                                              te(prev2_wordlen, bs="cr"), # +
                                              # te(log_freq, bs = "cr") +
                                              # te(prev_freq, bs = "cr") +
                                              # te(prev2_freq, bs = "cr"),
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
## create dataframe with equal entries across columns
llama3B_instruct_crit_new_data <- get_new_df(rc_rt_prev_crit.llama3B_instruct.data, 100, 2)

## predict the surprisal of words in critical regions using the full gam model 
llama3B_instruct_crit_predictions <- predict(llama3B_instruct_non_crit_model, 
                                             newdata=llama3B_instruct_crit_new_data,
                                             type="response",
                                             se.fit=TRUE)
## plot full model
plot_predictions(llama3B_instruct_crit_new_data, llama3B_instruct_crit_predictions)

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

### All model summary ----
model <- c("gpt2", "llama1b", "llama3b", "llama1b-instruct", "llama3b-instruct")
loglike <- c(gpt2_crit_delta_ll, llama1B_crit_delta_ll, llama3B_crit_delta_ll, llama1B_instruct_crit_delta_ll, llama3B_instruct_crit_delta_ll)
delta_ll <- data.frame(model = model, loglike = loglike)

ggplot(delta_ll,
       aes(x=model,
           y=loglike)) + 
  geom_point()+
  labs(x="Model",
       y="ΔLogLik")

## linear models ----
# for getting the predictive power
baseline_rt_regression = mean_rt ~ wordlen + prev_wordlen + prev2_wordlen + prev3_wordlen
full_rt_regression = mean_rt ~ surprisal + prev_surp + prev2_surp + prev3_surp + wordlen + prev_wordlen + prev2_wordlen + prev3_wordlen

# baseline_rt_mixed = mean_rt ~ wordlen + prev_wordlen + prev2_wordlen + (1|item_id)
# full_rt_regression = mean_rt ~ surprisal + prev_surp + prev2_surp + prev3_surp + wordlen + prev_wordlen + prev2_wordlen + prev3_wordlen

### GPT2 ----
gpt2_lm_base <- lm(baseline_rt_regression, data=rc_rt_prev.gpt2.data)
gpt2_lm_full <- lm(full_rt_regression, data=rc_rt_prev.gpt2.data)
AIC(gpt2_lm_base) # 8200.527
AIC(gpt2_lm_full) # 8197.384 <- more supported
anova(gpt2_lm_base, gpt2_lm_full)

# not giving a smooth line using predictions
# gpt2_linear_predictions <- data.frame(rt_pred = predict(gpt2_lm_full, rc_rt_prev.gpt2.data),
#                                       surprisal=rc_rt_prev.gpt2.data$surprisal)

gpt2_rt_linear_graph <- ggplot(rc_rt.gpt2.data,
                               aes(x=surprisal,
                                   y=mean_rt)) +
  geom_smooth(color="black", method="lm") +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
gpt2_rt_linear_graph


### Llama-3.2-1B ----
llama1B_lm_base <- lm(baseline_rt_regression, data=rc_rt_prev.llama1B.data)
llama1B_lm_full <- lm(full_rt_regression, data=rc_rt_prev.llama1B.data)
AIC(llama1B_lm_base) # 8200.527
AIC(llama1B_lm_full) # 8196.66 <- more supported
anova(llama1B_lm_base, llama1B_lm_full)

llama1B_rt_linear_graph <- ggplot(rc_rt.llama1B.data,
                                  aes(x=surprisal, 
                                      y=mean_rt)) +
  geom_smooth(color="black", method="lm") +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
llama1B_rt_linear_graph

### Llama-3.2-3B ----
llama3B_lm_base <- lm(baseline_rt_regression, data=rc_rt_prev.llama3B.data)
llama3B_lm_full <- lm(full_rt_regression, data=rc_rt_prev.llama3B.data)
AIC(llama3B_lm_base) # 8200.527
AIC(llama3B_lm_full) # 8191.496 <- more supported
anova(llama3B_lm_base, llama3B_lm_full)

llama3B_rt_linear_graph <- ggplot(rc_rt.llama3B.data,
                                  aes(x=surprisal, 
                                      y=mean_rt)) +
  geom_smooth(color="black", method="lm") +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
llama3B_rt_linear_graph

### Llama-3.2-1B-Instruct ----
llama1B_instruct_lm_base <- lm(baseline_rt_regression, data=rc_rt_prev.llama1B_instruct.data)
llama1B_instruct_lm_full <- lm(full_rt_regression, data=rc_rt_prev.llama1B_instruct.data)
AIC(llama1B_instruct_lm_base) # 8200.527
AIC(llama1B_instruct_lm_full) # 8200.167
anova(llama1B_instruct_lm_base, llama1B_instruct_lm_full)

llama1B_instruct_rt_linear_graph <- ggplot(rc_rt.llama1B_instruct.data,
                                           aes(x=surprisal,
                                               y=mean_rt)) +
  geom_smooth(color="black", method="lm") +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
llama1B_instruct_rt_linear_graph

### Llama-3.2-3B-Instruct ----
llama3B_instruct_lm_base <- lm(baseline_rt_regression, data=rc_rt_prev.llama3B_instruct.data)
llama3B_instruct_lm_full <- lm(full_rt_regression, data=rc_rt_prev.llama3B_instruct.data)
AIC(llama3B_instruct_lm_base) # 8200.527
AIC(llama3B_instruct_lm_full) # 8192.172 <- more supported
anova(llama3B_instruct_lm_base, llama3B_instruct_lm_full)

llama3B_instruct_rt_linear_graph <- ggplot(rc_rt.llama3B_instruct.data,
                                           aes(x=surprisal,
                                               y=mean_rt)) +
  geom_smooth(color="black", method="lm") +
  labs(y="Mean reading time (ms)",
       x="Surprisal")
llama3B_instruct_rt_linear_graph

