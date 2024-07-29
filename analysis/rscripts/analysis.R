library(lme4)
library(dplyr)
library(emmeans)
library(tidyverse)
library(ggplot2)
library(tidytext)
library(RColorBrewer)
library(stringr)


theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 


# 1. Data ----
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

## Comprehension ----
comprehension.gpt4o.data <- read.csv("../../data/comprehension-gpt-4o_annotate.csv", header=TRUE) %>% 
  na.omit()
comprehension.gpt4.data <- read.csv("../../data/comprehension-gpt-4-annotate.csv", header=TRUE) %>% 
  na.omit()
comprehension.gpt35.data <- read.csv("../../data/comprehension-gpt-3.5-turbo_annotate.csv", header=TRUE) %>% 
  na.omit()

comprehension_data <- bind_rows(lst(comprehension.gpt4o.data, comprehension.gpt4.data, comprehension.gpt35.data), .id="model") %>% 
  filter(verb_type %in% c("IC_High", "nonIC_High")) %>% 
  mutate(model = case_when(model == "comprehension.gpt4o.data" ~ "gpt-4o",
                           model == "comprehension.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "comprehension.gpt4.data" ~ "gpt-4"),
         explanation = if_else(explanation == "no", "no", "yes")) %>% 
  select(model, sentence_type, explanation, item_id)

comprehension_means <- comprehension_data %>% 
  mutate(explanation_numerical = ifelse(explanation=="no", 0, 1)) %>% 
  group_by(model, sentence_type) %>% 
  summarize(Mean = mean(explanation_numerical),
            CILow = ci.low(explanation_numerical),
            CIHigh = ci.high(explanation_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

## RC ----
rc.gpt4o.data <- read.csv("../../data/rc-gpt-4o.csv", header=TRUE) %>% 
  na.omit()
rc.gpt4.data <- read.csv("../../data/rc-gpt-4.csv", header=TRUE) %>% 
  na.omit()
rc.gpt35.data <- read.csv("../../data/rc-gpt-3.5-turbo.csv", header=TRUE) %>% 
  na.omit()

rc_data <- bind_rows(lst(rc.gpt4o.data, rc.gpt4.data, rc.gpt35.data), .id="model") %>% 
  mutate(model = case_when(model == "rc.gpt4o.data" ~ "gpt-4o",
                           model == "rc.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "rc.gpt4.data" ~ "gpt-4"),
         np_order = if_else(item_id < 11, "high_sg", "high_pl"),
         answer_choice = as.numeric(gsub("\\D", "", answer)),
         answer_option = strsplit(answer_option, ", "),
         option_1 = gsub("[^a-zA-Z]", "", lapply(answer_option, `[[`, 1)),
         option_2 = gsub("[^a-zA-Z]", "",lapply(answer_option, `[[`, 2)),
         answer = ifelse(answer_choice == 1, option_1, option_2),
         sg_pl = if_else(answer %in% c("have", "were", "are"), "pl", "sg"),
         attachment = case_when(np_order == "high_sg" & sg_pl == "sg" ~ "high",
                                np_order == "high_sg" & sg_pl == "pl" ~ "low",
                                np_order == "high_pl" & sg_pl == "pl" ~ "high",
                                np_order == "high_pl" & sg_pl == "sg" ~ "low")) %>% 
  select(model, item_id,sentence_type,attachment)

rc_means <- rc_data %>% 
  mutate(attachment_numerical = ifelse(attachment=="high", 1, 0)) %>%
  group_by(model, sentence_type) %>% 
  summarize(Mean = mean(attachment_numerical),
            CILow = ci.low(attachment_numerical),
            CIHigh = ci.high(attachment_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

## combined ----
comprehension_rc_data <- merge(comprehension_data, rc_data, by=c("model","item_id","sentence_type")) %>% 
  mutate(exp_verb_group = case_when(sentence_type == "IC" & explanation == "yes" ~ "IC\n-exp",
                                    sentence_type == "IC" & explanation == "no" ~ "IC\n-nonexp",
                                    sentence_type == "nonIC" & explanation == "yes" ~ "nonIC\n-exp",
                                    sentence_type == "nonIC" & explanation == "no" ~ "nonIC\n-nonexp"))

comprehension_rc_mean <- comprehension_rc_data %>% 
  mutate(attachment_numerical = ifelse(attachment=="high", 1, 0),
         explanation = ifelse(explanation == "yes", "exp", "non-exp")) %>%
  group_by(model, explanation) %>% 
  summarize(Mean = mean(attachment_numerical),
            CILow = ci.low(attachment_numerical),
            CIHigh = ci.high(attachment_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)
  

comprehension_rc_verb_mean <- comprehension_rc_data %>% 
  mutate(attachment_numerical = ifelse(attachment=="high", 1, 0),
         explanation = ifelse(explanation == "yes", "exp", "non-exp")) %>%
  group_by(model, exp_verb_group) %>% 
  summarize(Mean = mean(attachment_numerical),
            CILow = ci.low(attachment_numerical),
            CIHigh = ci.high(attachment_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)
  


# 2. Plot ----
## Comprehension ----
comprehension_graph <- ggplot(comprehension_means,
                              aes(x=sentence_type,y=Mean))+
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax),width=.2,  show.legend = FALSE)+
  theme_bw() +
  labs(y = "Proportion of explanation answer",
       x = "Verb Type") +
  facet_wrap(. ~ model)
comprehension_graph
ggsave(comprehension_graph, file="../graphs/comprehension_pilot.pdf", width=7, height=4)

## RC ----
rc_graph <- ggplot(rc_means,
                              aes(x=sentence_type,y=Mean))+
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax),width=.2,  show.legend = FALSE)+
  theme_bw() +
  labs(y = "Proportion of high attachment",
       x = "Verb Type") +
  facet_wrap(. ~ model)
rc_graph
ggsave(rc_graph, file="../graphs/rc_pilot.pdf", width=7, height=4)

## combined ----
comprehension_rc_graph <- ggplot(comprehension_rc_mean,
                                 aes(x=explanation,y=Mean))+
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax),width=.2,  show.legend = FALSE)+
  theme_bw() +
  labs(y = "Proportion of high attachment",
       x = "RC Type") +
  facet_wrap(. ~ model)
comprehension_rc_graph
ggsave(comprehension_rc_graph, file="../graphs/comprehension_rc_pilot.pdf", width=7, height=4)

comprehension_rc_verb_graph <- 
  ggplot(comprehension_rc_verb_mean,
         aes(x=exp_verb_group,y=Mean))+
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax),width=.2,  show.legend = FALSE)+
  theme_bw() +
  labs(y = "Proportion of high attachment",
       x = "RC Type and Verb Type") +
  # theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=0.5))+
  facet_wrap(. ~ model)
comprehension_rc_verb_graph
ggsave(comprehension_rc_graph, file="../graphs/comprehension_rc_pilot.pdf", width=7, height=4)
