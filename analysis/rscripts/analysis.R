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

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
## Comprehension ----
### original prompt ----
comprehension.gpt4o.data <- read.csv("../../data/comprehension/comprehension-gpt-4o_annotate.csv", header=TRUE) %>% 
  na.omit()
comprehension.gpt4.data <- read.csv("../../data/comprehension/comprehension-gpt-4-annotate.csv", header=TRUE) %>% 
  na.omit()
comprehension.gpt35.data <- read.csv("../../data/comprehension/comprehension-gpt-3.5-turbo_annotate.csv", header=TRUE) %>% 
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


### alternative prompt ----
comprehension_alt.gpt4o.data <- read.csv("../../data/comprehension_alt/comprehension_alt-gpt-4o.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no"))
comprehension_alt.gpt4.data <- read.csv("../../data/comprehension_alt/comprehension_alt-gpt-4.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no"))
comprehension_alt.gpt35.data <- read.csv("../../data/comprehension_alt/comprehension_alt-gpt-3.5-turbo.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no"))

comprehension_alt_data <- bind_rows(lst(comprehension_alt.gpt4o.data, comprehension_alt.gpt4.data, comprehension_alt.gpt35.data), .id="model") %>% 
  filter(verb_type %in% c("IC_High", "nonIC_High")) %>% 
  mutate(model = case_when(model == "comprehension_alt.gpt4o.data" ~ "gpt-4o",
                           model == "comprehension_alt.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "comprehension_alt.gpt4.data" ~ "gpt-4"),
         explanation = if_else(explanation == "no", "no", "yes")) %>% 
  select(model, sentence_type, explanation, item_id)

comprehension_alt_means <- comprehension_alt_data %>% 
  mutate(explanation_numerical = ifelse(explanation=="no", 0, 1)) %>% 
  group_by(model, sentence_type) %>% 
  summarize(Mean = mean(explanation_numerical),
            CILow = ci.low(explanation_numerical),
            CIHigh = ci.high(explanation_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

## RC ----
### original prompt ----
# the model is presented with the full context
rc.gpt4o.data <- read.csv("../../data/rc/rc-gpt-4o.csv", header=TRUE) %>% 
  na.omit()
rc.gpt4.data <- read.csv("../../data/rc/rc-gpt-4.csv", header=TRUE) %>% 
  na.omit()
rc.gpt35.data <- read.csv("../../data/rc/rc-gpt-3.5-turbo.csv", header=TRUE) %>% 
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

### alternative prompt ----
rc_alt.gpt4o.data <- read.csv("../../data/rc_alt/rc_alt-gpt-4o.csv", header=TRUE) %>% 
  na.omit() 
rc_alt.gpt4.data <- read.csv("../../data/rc_alt/rc_alt-gpt-4.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(answer=as.character(answer))
rc_alt.gpt35.data <- read.csv("../../data/rc_alt/rc_alt-gpt-3.5-turbo.csv", header=TRUE) %>% 
  na.omit()

rc_alt_data <- bind_rows(lst(rc_alt.gpt4o.data, rc_alt.gpt4.data, rc_alt.gpt35.data), .id="model") %>% 
  mutate(model = case_when(model == "rc_alt.gpt4o.data" ~ "gpt-4o",
                           model == "rc_alt.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "rc_alt.gpt4.data" ~ "gpt-4"),
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

rc_alt_means <- rc_alt_data %>% 
  mutate(attachment_numerical = ifelse(attachment=="high", 1, 0)) %>%
  group_by(model, sentence_type) %>% 
  summarize(Mean = mean(attachment_numerical),
            CILow = ci.low(attachment_numerical),
            CIHigh = ci.high(attachment_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

## combined ----
### with original prompt ----
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

exp_rc_verb_data <- merge(comprehension_data, rc_data, by=c("model","item_id","sentence_type")) %>% 
  mutate(rc_verb_group = case_when(sentence_type == "IC" & attachment == "low" ~ "IC\n-low",
                                    sentence_type == "IC" & attachment == "high" ~ "IC\n-high",
                                    sentence_type == "nonIC" & attachment == "low" ~ "nonIC\n-low",
                                    sentence_type == "nonIC" & attachment == "high" ~ "nonIC\n-high"))

exp_rc_verb_mean <- exp_rc_verb_data %>% 
  mutate(explanation_numerical = ifelse(explanation=="yes", 1, 0)) %>%
  group_by(model, rc_verb_group) %>% 
  summarize(Mean = mean(explanation_numerical),
            CILow = ci.low(explanation_numerical),
            CIHigh = ci.high(explanation_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

### with alternative prompt ----
comp_alt_rc_data <- merge(comprehension_alt_data, rc_data, by=c("model","item_id","sentence_type")) %>%
  mutate(exp_verb_group = case_when(sentence_type == "IC" & explanation == "yes" ~ "IC\n-exp",
                                    sentence_type == "IC" & explanation == "no" ~ "IC\n-nonexp",
                                    sentence_type == "nonIC" & explanation == "yes" ~ "nonIC\n-exp",
                                    sentence_type == "nonIC" & explanation == "no" ~ "nonIC\n-nonexp"))

comp_alt_rc_mean <- comp_alt_rc_data %>% 
  mutate(attachment_numerical = ifelse(attachment=="high", 1, 0),
         explanation = ifelse(explanation == "yes", "exp", "non-exp")) %>%
  group_by(model, explanation) %>% 
  summarize(Mean = mean(attachment_numerical),
            CILow = ci.low(attachment_numerical),
            CIHigh = ci.high(attachment_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

comp_alt_rc_verb_mean <- comp_alt_rc_data %>% 
  mutate(attachment_numerical = ifelse(attachment=="high", 1, 0),
         explanation = ifelse(explanation == "yes", "exp", "non-exp")) %>%
  group_by(model, exp_verb_group) %>% 
  summarize(Mean = mean(attachment_numerical),
            CILow = ci.low(attachment_numerical),
            CIHigh = ci.high(attachment_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

exp_rc_verb_alt_data <- merge(comprehension_alt_data, rc_data, by=c("model","item_id","sentence_type")) %>% 
  mutate(rc_verb_group = case_when(sentence_type == "IC" & attachment == "low" ~ "IC\n-low",
                                   sentence_type == "IC" & attachment == "high" ~ "IC\n-high",
                                   sentence_type == "nonIC" & attachment == "low" ~ "nonIC\n-low",
                                   sentence_type == "nonIC" & attachment == "high" ~ "nonIC\n-high"))

exp_rc_verb_alt_mean <- exp_rc_verb_alt_data %>% 
  mutate(explanation_numerical = ifelse(explanation=="yes", 1, 0)) %>%
  group_by(model, rc_verb_group) %>% 
  summarize(Mean = mean(explanation_numerical),
            CILow = ci.low(explanation_numerical),
            CIHigh = ci.high(explanation_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)


## Pronoun ----
### free prompt ----
pronoun.free.gpt4o.data <- read.csv("../../data/pronoun_free/pronoun_free-gpt-4o_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  filter(pronoun %in% c("yes", "no") & object %in% c("yes", "no"))
pronoun.free.gpt4.data <- read.csv("../../data/pronoun_free/pronoun_free-gpt-4_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  filter(pronoun %in% c("yes", "no") & object %in% c("yes", "no"))
pronoun.free.gpt35.data <- read.csv("../../data/pronoun_free/pronoun_free-gpt-3.5-turbo_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  filter(pronoun %in% c("yes", "no") & object %in% c("yes", "no"))

pronoun_free_data <- bind_rows(lst(pronoun.free.gpt4o.data, pronoun.free.gpt4.data, pronoun.free.gpt35.data), .id="model") %>% 
  mutate(model = case_when(model == "pronoun.free.gpt4o.data" ~ "gpt-4o",
                           model == "pronoun.free.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "pronoun.free.gpt4.data" ~ "gpt-4"),
         pronoun_numerical = ifelse(pronoun == "yes", 1, 0),
         explanation_numerical = ifelse(explanation == "yes", 1, 0),
         reference = if_else(object == "yes", "object", "subject"),
         RC_type = if_else(grepl("nonexp", condition), "nonexp", "exp"))

### pronoun prompt ----
pronoun.pro.gpt4o.data <- read.csv("../../data/pronoun_pro/pronoun_pro-gpt-4o_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  filter(object %in% c("yes", "no"))
pronoun.pro.gpt4.data <- read.csv("../../data/pronoun_pro/pronoun_pro-gpt-4_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  filter(object %in% c("yes", "no"))
pronoun.pro.gpt35.data <- read.csv("../../data/pronoun_pro/pronoun_pro-gpt-3.5-turbo_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  filter(object %in% c("yes", "no"))

pronoun_pro_data <- bind_rows(lst(pronoun.pro.gpt4o.data, pronoun.pro.gpt4.data, pronoun.pro.gpt35.data), .id="model") %>% 
  mutate(model = case_when(model == "pronoun.pro.gpt4o.data" ~ "gpt-4o",
                           model == "pronoun.pro.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "pronoun.pro.gpt4.data" ~ "gpt-4"),
         object_numerical = ifelse(object == "yes", 1, 0),
         explanation_numerical = ifelse(explanation == "yes", 1, 0),
         RC_type = if_else(grepl("nonexp", condition), "nonexp", "exp"))

### both prompts combined ----
pronoun_data <- bind_rows(lst(pronoun_pro_data, pronoun_free_data), .id="task") %>% 
  mutate(task = if_else(task == "pronoun_free_data", "free", "pronoun"))

#### the use of pronoun in the free prompt
pronoun_use_mean <- pronoun_free_data %>% 
  select(model, RC_type, condition, sentence, reference,
         pronoun, pronoun_numerical) %>% 
  group_by(model, RC_type, reference) %>% 
  summarize(Mean = mean(pronoun_numerical),
            CILow = ci.low(pronoun_numerical),
            CIHigh = ci.high(pronoun_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

#### object mentions in both prompts
object_mean <- pronoun_data %>% 
  select(task, RC_type, model, condition, sentence, 
         object, object_numerical) %>% 
  group_by(task, RC_type, model, condition) %>% 
  summarize(Mean = mean(object_numerical),
            CILow = ci.low(object_numerical),
            CIHigh = ci.high(object_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)
  
#### the proportion of explanation continuation in both prompts
explanation_mean <- pronoun_data %>% 
  select(task, RC_type, model, condition, sentence, 
         explanation, explanation_numerical) %>% 
  group_by(RC_type, model) %>% 
  summarize(Mean = mean(explanation_numerical),
            CILow = ci.low(explanation_numerical),
            CIHigh = ci.high(explanation_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)


# 2. Plot ----
## Comprehension ----
comprehension_graph <- ggplot(comprehension_means,
                              aes(x=sentence_type,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of explanation answers",
       x = "Verb Type") +
  facet_wrap(. ~ model)
comprehension_graph
ggsave(comprehension_graph, file="../graphs/comprehension_pilot.pdf", width=7, height=4)

## comp. alt ----
comprehension_alt_graph <- ggplot(comprehension_alt_means,
                              aes(x=sentence_type,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of explanation answers",
       x = "Verb Type") +
  facet_wrap(. ~ model) +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10))
comprehension_alt_graph
ggsave(comprehension_alt_graph, file="../graphs/comprehension_alt_pilot_1.pdf", width=5, height=4)

## RC ----
rc_graph <- ggplot(rc_means,
                              aes(x=sentence_type,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of high attachment",
       x = "Verb Type") +
  facet_wrap(. ~ model)
rc_graph
ggsave(rc_graph, file="../graphs/rc_pilot.jpeg", width=5, height=4)

## RC alt ----
rc_alt_graph <- ggplot(rc_alt_means,
                   aes(x=sentence_type,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of high attachment",
       x = "Verb Type") +
  facet_wrap(. ~ model)
rc_alt_graph
ggsave(rc_alt_graph, file="../graphs/rc_alt_pilot.pdf", width=7, height=4)

## combined ----
comprehension_rc_graph <- ggplot(comprehension_rc_mean,
                                 aes(x=explanation,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of high attachment",
       x = "RC Type") +
  facet_wrap(. ~ model)
comprehension_rc_graph
ggsave(comprehension_rc_graph, file="../graphs/comprehension_rc_pilot.pdf", width=7, height=4)

comprehension_rc_verb_graph <- ggplot(comprehension_rc_verb_mean,
                                      aes(x=exp_verb_group,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of high attachment",
       x = "RC Type and Verb Type") +
  # theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=0.5)) +
  facet_wrap(. ~ model)
comprehension_rc_verb_graph
ggsave(comprehension_rc_verb_graph, file="../graphs/comprehension_rc_verb_pilot.pdf", width=7, height=4)

exp_rc_verb_graph <- ggplot(exp_rc_verb_mean,
                            aes(x=rc_verb_group,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE)+
  theme_bw() +
  labs(y = "Proportion of explanation answer",
       x = "Attachment height and Verb Type") +
  # theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=0.5))+
  facet_wrap(. ~ model)
exp_rc_verb_graph
ggsave(exp_rc_verb_graph, file="../graphs/exp_rc_verb_pilot.pdf", width=7, height=4)

## combined with comp. alt ----
comp_alt_rc_graph <- ggplot(comp_alt_rc_mean,
                                 aes(x=explanation,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of high attachment",
       x = "RC Type") +
  facet_wrap(. ~ model)
comp_alt_rc_graph
ggsave(comp_alt_rc_graph, file="../graphs/comprehension_alt_rc_pilot.pdf", width=7, height=4)

comp_alt_rc_verb_graph <- ggplot(comp_alt_rc_verb_mean,
                                 aes(x=exp_verb_group,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of high attachment",
       x = "RC Type and Verb Type") +
  # theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=0.5))+
  facet_wrap(. ~ model)
comp_alt_rc_verb_graph
ggsave(comp_alt_rc_verb_graph, file="../graphs/comprehension_alt_rc_verb_pilot.pdf", width=7, height=4)

exp_rc_verb_alt_graph <- ggplot(exp_rc_verb_alt_mean,
                                aes(x=rc_verb_group,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of explanation answer",
       x = "Attachment height and Verb Type") +
  # theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=0.5)) +
  facet_wrap(. ~ model)
exp_rc_verb_alt_graph
ggsave(exp_rc_verb_alt_graph, file="../graphs/exp_rc_verb_alt_pilot.pdf", width=7, height=4)


## Pronoun ----
### explanation continuation ----
exp_graph <- ggplot(explanation_mean,
                    aes(x=RC_type,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),width=.2,
                show.legend = FALSE)+
  theme_bw() +
  labs(y = "Proportion of explanation continuation",
       x = "RC Type") +
  facet_wrap(. ~ model)
exp_graph
ggsave(exp_graph, file="../graphs/explanation_pilot.pdf", width=7, height=4)

### object mentions in free prompt ----
object_free_graph <- ggplot(object_mean %>% 
                              filter(task == "free"),
                            aes(x=RC_type,y=Mean)) +
  geom_bar(stat="identity", width=0.8, alpha=0.7) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),width=.2,
                show.legend = FALSE)+
  theme_bw() +
  labs(y = "Proportion of object mentions",
       x = "RC Type") +
  facet_wrap(. ~ model)
object_free_graph
ggsave(object_free_graph, file="../graphs/object_free_pilot.pdf", width=7, height=4)

### pronoun use in free prompt ----
pronoun_free_graph <- ggplot(pronoun_use_mean,
                             aes(x=RC_type,y=Mean,
                                 fill=reference)) +
  geom_bar(position=position_dodge(), stat="identity",
           alpha=0.7, colour="black", size=0.3) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                position=position_dodge(width = 0.9),
                width=.2,
                show.legend = FALSE)+
  theme_bw() +
  labs(y = "Proportion of pronoun uses",
       x = "RC Type") +
  scale_fill_manual(labels = c("Object", "Subject"), values = c("white", "black")) +
  guides(fill=guide_legend(title="Pronoun\nreference")) +
  facet_wrap(. ~ model)
pronoun_free_graph
ggsave(pronoun_free_graph, file="../graphs/pronoun_free_pilot.pdf", width=7, height=4)

### object use in both prompts ----
object_graph <- ggplot(object_mean,
                       aes(x=RC_type,y=Mean,
                           fill=task)) +
  geom_bar(position=position_dodge(), stat="identity",
           alpha=0.7, colour="black", size=0.3) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                position=position_dodge(width = 0.9),
                width=.2,
                show.legend = FALSE)+
  theme_bw() +
  labs(y = "Proportion of object mentions",
       x = "RC Type") +
  scale_fill_manual(labels = c("Full-stop", "Pronoun"), values = c("white", "black")) +
  guides(fill=guide_legend(title="Prompt")) +
  facet_wrap(. ~ model)
object_graph
ggsave(object_graph, file="../graphs/object_pilot.pdf", width=7, height=4)
