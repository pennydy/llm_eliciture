library(lme4)
library(lmerTest)
library(brms)
library(dplyr)
library(emmeans)
library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggpattern)
library(tidytext)
library(RColorBrewer)
library(stringr)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
## gpt4o ----
comprehension_and_so.gpt4o.data <- read.csv("../../../data/comprehension_nonIC_and_so_1/comprehension_nonIC_and_so_1-gpt-4o_2.csv", header=TRUE) %>% 
  na.omit()

comprehension_and_so.gpt4o.data <- comprehension_and_so.gpt4o.data %>% 
  rename(rc_type = "continuation_type") %>% 
  mutate(answer_choice = as.numeric(gsub("\\D", "", answer)),
         answer_option = strsplit(answer_option, ", "),
         option_1 = gsub("[^a-zA-Z]", "", lapply(answer_option, `[[`, 1)),
         option_2 = gsub("[^a-zA-Z]", "",lapply(answer_option, `[[`, 2)),
         connective = ifelse(answer_choice == 1, option_1, option_2),
         answer = toString(answer))

## gpt4 ----
comprehension_and_so.gpt4.data <- read.csv("../../../data/comprehension_nonIC_and_so_1/comprehension_nonIC_and_so_1-gpt-4_2.csv", header=TRUE) %>% 
  na.omit()

comprehension_and_so.gpt4.data <- comprehension_and_so.gpt4.data %>%
  rename(rc_type = "continuation_type") %>% 
  mutate(answer_choice = as.numeric(gsub("\\D", "", answer)),
         answer_option = strsplit(answer_option, ", "),
         option_1 = gsub("[^a-zA-Z]", "", lapply(answer_option, `[[`, 1)),
         option_2 = gsub("[^a-zA-Z]", "",lapply(answer_option, `[[`, 2)),
         connective = ifelse(answer_choice == 1, option_1, option_2),
         answer = toString(answer))

## gpt3.5-turbo ----
comprehension_and_so.gpt35.data <- read.csv("../../../data/comprehension_nonIC_and_so_1/comprehension_nonIC_and_so_1-gpt-3.5-turbo_2.csv", header=TRUE) %>% 
  na.omit()

comprehension_and_so.gpt35.data <- comprehension_and_so.gpt35.data %>% 
  rename(rc_type = "continuation_type") %>% 
  mutate(answer_choice = as.numeric(gsub("\\D", "", answer)),
         answer_option = strsplit(answer_option, ", "),
         option_1 = gsub("[^a-zA-Z]", "", lapply(answer_option, `[[`, 1)),
         option_2 = gsub("[^a-zA-Z]", "",lapply(answer_option, `[[`, 2)),
         connective = ifelse(answer_choice == 1, option_1, option_2),
         answer = toString(answer))

## combined ----
comprehension_and_so.data <- bind_rows(lst(comprehension_and_so.gpt4o.data, comprehension_and_so.gpt4.data, comprehension_and_so.gpt35.data), .id="model") %>% 
  mutate(model = case_when(model == "comprehension_and_so.gpt4o.data" ~ "gpt-4o",
                           model == "comprehension_and_so.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "comprehension_and_so.gpt4.data" ~ "gpt-4")) %>% 
  select(model, sent_id, item_id, verb, verb_type, rc_type,connective)

comprehension_and_so_means <- comprehension_and_so.data %>% 
  mutate(connective_numerical = ifelse(connective=="because", 1, 0)) %>%
  group_by(model, verb_type, rc_type) %>% 
  summarize(Mean = mean(connective_numerical),
            CILow = ci.low(connective_numerical),
            CIHigh = ci.high(connective_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

# 2. Plot ----
## line graph for IC verbs mean ----
comprehension_ic_line_graph <- ggplot(comprehension_and_so_means %>% 
                                        filter(verb_type == "IC"),
                                      aes(x=rc_type,y=Mean)) +
  geom_point(stat="identity",
             alpha=0.7,
             size=2) +
  geom_hline(yintercept=0.5, linetype="dashed", color = "grey") +
  geom_line(aes(group=1),linetype="dotted") +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  facet_grid(. ~ model) + 
  labs(y = "Proportion of \"becuase\" ",
       x = "RC type") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 12),
        axis.title.y = element_text(size = 12))
comprehension_ic_line_graph
ggsave(comprehension_ic_line_graph, file="../graphs/comprehension_ic_line_graph.pdf", width=8, height=3)

## bar graph for all verbs ----
comprehension_and_so_graph <- ggplot(comprehension_and_so_means,
                       aes(x=verb_type,y=Mean,
                           fill=verb_type,
                           pattern=rc_type)) +
  geom_bar_pattern(
    position = "dodge",
    stat="identity",
    pattern_angle = 45,
    pattern_spacing = 0.02,
    pattern_fill="black",
    pattern_alpha=0.4,
    alpha=0.7) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                position=position_dodge(width=0.8),
                width=.2, 
                show.legend = FALSE) +
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  theme_bw() +
  labs(y = "Proportion of because",
       x = "Verb Type") +
  facet_wrap(. ~ model) +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10)) +
  scale_fill_brewer(palette = "Dark2",guide="none")
comprehension_and_so_graph
ggsave(comprehension_and_so_graph, file="../graphs/comprehension_and_so_graph_2.pdf", width=7, height=4)

# 3. Analysis ----
## gpt-3.5-turbo ----
comprehension_and_so.gpt35.data <- comprehension_and_so.gpt35.data %>% 
  filter(verb_type == "IC") %>% 
  mutate(rc_type = fct_relevel(rc_type, ref="exp"),
         connective = fct_relevel(connective, ref="because"))
gpt35 <- glmer(connective ~ rc_type + (1+rc_type|item_id),
               family = "binomial",
               data=comprehension_and_so.gpt35.data)
summary(gpt35)

gpt35_connective_bayesian_default_prior <- brm(connective ~ rc_type + (1+rc_type|item_id),
                                               family = "bernoulli",
                                              data=comprehension_and_so.gpt35.data,
                                              iter=8000,
                                              warmup = 4000,
                                              chains=4,
                                              cores=4,
                                              control=list(max_treedepth = 15, adapt_delta = 0.99),
                                              file="../cache/gpt35_connective_bayesian_default_prior",
                                              seed=1024)
summary(gpt35_connective_bayesian_default_prior)

## gpt-4 ----
comprehension_and_so.gpt4.data <- comprehension_and_so.gpt4.data %>% 
  filter(verb_type == "IC") %>% 
  mutate(rc_type = fct_relevel(rc_type,ref="exp"),
         connective = fct_relevel(connective, ref="because"))
gpt4 <- glmer(connective ~ rc_type + (1+rc_type|item_id),
               family = "binomial",
               data=comprehension_and_so.gpt4.data)
summary(gpt4)

gpt4_connective_bayesian_default_prior <- brm(connective ~ rc_type + (1+rc_type|item_id),
                                               family = "bernoulli",
                                               data=comprehension_and_so.gpt4.data,
                                               iter=8000,
                                               warmup = 4000,
                                               chains=4,
                                               cores=4,
                                               control=list(max_treedepth = 15, adapt_delta = 0.99),
                                               file="../cache/gpt4_connective_bayesian_default_prior",
                                               seed=1024)
summary(gpt4_connective_bayesian_default_prior)

## gpt-4o ----
comprehension_and_so.gpt4o.data <- comprehension_and_so.gpt4o.data %>% 
  filter(verb_type == "IC") %>% 
  mutate(rc_type = fct_relevel(rc_type, ref="exp"),
         connective = fct_relevel(connective, ref="because"))
# contrasts(comprehension_and_so.gpt4o.data$rc_type)=contr.sum(2)
# contrasts(comprehension_and_so.gpt4o.data$verb_type)=contr.sum(2)
gpt4o <- glmer(connective ~ rc_type + (1+rc_type|item_id),
              family = "binomial",
              data=comprehension_and_so.gpt4o.data)
summary(gpt4o)

gpt4o_connective_bayesian_default_prior <- brm(connective ~ rc_type + (1+rc_type|item_id),
                                              family = "bernoulli",
                                              data=comprehension_and_so.gpt4o.data,
                                              iter=8000,
                                              warmup = 4000,
                                              chains=4,
                                              cores=4,
                                              control=list(max_treedepth = 15, adapt_delta = 0.99),
                                              file="../cache/gpt4o_connective_bayesian_default_prior",
                                              seed=1024)
summary(gpt4o_connective_bayesian_default_prior)
