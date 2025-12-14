library(lme4)
library(lmerTest)
library(dplyr)
library(emmeans)
library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggpattern)
library(tidytext)
library(RColorBrewer)
library(stringr)
library(grid)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")


# 1. Data ----
# files with _why suffix include the log prob of "why" and the final period
# files without the suffix include the log prob of only "why"
## Llama3.2-1B ----
comprehension.llama1B.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-1B_2.csv", header=TRUE) %>%
  na.omit()

comprehension.llama1B.data <- comprehension.llama1B.data %>% 
  rename(rc_type = "continuation_type") %>% 
  select(item_id, verb, verb_type, rc_type, critical_prob) %>%
  pivot_wider(names_from = rc_type, values_from = critical_prob) %>%
  mutate(logodds = exp-nonexp,
         exp_prob = exp(exp)/1+exp(exp),
         nonexp_prob = exp(nonexp)/1+exp(nonexp))

## Llama3.2-1B-Instruct ----
comprehension.llama1B.instruct.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-1B-Instruct_2.csv", header=TRUE) %>%
  na.omit()

comprehension.llama1B.instruct.data <- comprehension.llama1B.instruct.data %>% 
  rename(rc_type = "continuation_type") %>% 
  select(item_id, verb, verb_type, rc_type, critical_prob) %>%
  pivot_wider(names_from = rc_type, values_from = critical_prob) %>%
  mutate(logodds = exp-nonexp,
         exp_prob = exp(exp)/1+exp(exp),
         nonexp_prob = exp(nonexp)/1+exp(nonexp))

## Llama3.2-3B ----
comprehension.llama3B.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-3B_2.csv", header=TRUE) %>%
  na.omit()

comprehension.llama3B.data <- comprehension.llama3B.data %>% 
  rename(rc_type = "continuation_type") %>%
  select(item_id, verb, verb_type, rc_type, critical_prob) %>%
  pivot_wider(names_from = rc_type, values_from = critical_prob) %>%
  mutate(logodds = exp-nonexp,
         exp_prob = exp(exp)/1+exp(exp),
         nonexp_prob = exp(nonexp)/1+exp(nonexp))

## Llama3.2-3B-Instruct ----
comprehension.llama3B.instruct.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-3B-Instruct_2.csv", header=TRUE) %>%
  na.omit()

comprehension.llama3B.instruct.data <- comprehension.llama3B.instruct.data %>% 
  rename(rc_type = "continuation_type") %>%
  select(item_id, verb, verb_type, rc_type, critical_prob) %>%
  pivot_wider(names_from = rc_type, values_from = critical_prob) %>%
  mutate(logodds = exp-nonexp,
         exp_prob = exp(exp)/1+exp(exp),
         nonexp_prob = exp(nonexp)/1+exp(nonexp))

## gpt2 ----
comprehension.gpt2.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_gpt2_2.csv", header=TRUE) %>%
  na.omit()

comprehension.gpt2.data <- comprehension.gpt2.data %>% 
  rename(rc_type = "continuation_type") %>%
  select(item_id, verb, verb_type, rc_type, critical_prob) %>%
  pivot_wider(names_from = rc_type, values_from = critical_prob) %>%
  mutate(logodds = exp-nonexp,
         exp_prob = exp(exp)/1+exp(exp),
         nonexp_prob = exp(nonexp)/1+exp(nonexp))

## all models ----
comprehension.data <- bind_rows(lst(comprehension.llama1B.data,comprehension.llama1B.instruct.data,comprehension.llama3B.data,comprehension.llama3B.instruct.data,comprehension.gpt2.data), .id="model") %>% 
  mutate(model=case_when(model == "comprehension.llama1B.data" ~ "Llama3.2-1B",
                         model == "comprehension.llama1B.instruct.data" ~ "Llama3.2-1B-Instruct",
                         model == "comprehension.llama3B.data" ~ "Llama3.2-3B",
                         model == "comprehension.llama3B.instruct.data" ~ "Llama3.2-3B-Instruct",
                         model == "comprehension.gpt2.data" ~ "GPT2"))

## all models (mean) ----
comprehension_mean <- comprehension.data %>% 
  select(-c(logodds, exp_prob, nonexp_prob)) %>% 
  pivot_longer(cols=c(exp, nonexp),
               names_to = "rc_type",
               values_to = "critical_region_logprob") %>% 
  group_by(model,verb_type, rc_type) %>% 
  summarize(Mean = mean(critical_region_logprob),
            CILow = ci.low(critical_region_logprob),
            CIHigh = ci.high(critical_region_logprob)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)
  
# 2. Plot ----
## all models ----
# desnity plot doesn't really make sense -> predicting the log prob of the same sentence
### density of log odds ----
# comprehension_graph <- ggplot(comprehension.data,
#                               aes(x=logodds,
#                                   fill=verb_type)) +
#   geom_density(alpha=0.6) +
#   theme_bw() +
#   scale_fill_brewer(palette = "Dark2") +
#   geom_vline(xintercept = 0, linetype="dotted") +
#   labs(fill = "Verb type") +
#   coord_cartesian(clip="off") +
#   facet_grid(model ~ .) +
#   theme(legend.position = "top",
#         axis.text.x = element_text(size = 10),
#         axis.text.y = element_text(size = 10)) +
#   labs(y = "density",
#        x = "log odds (exp vs. nonexp)")
# comprehension_graph
# ggsave(comprehension_graph, file="../graphs/comprehension_nonIC_sent_why.pdf", width=8, height=4)

### bar graph for means ----
comprehension_bar_graph <- ggplot(comprehension_mean,
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
  labs(y = "Mean log probability of 'why'",
       x = "Verb Type") +
  facet_grid(. ~ model) +
  theme(legend.position = "top",
        axis.title.x = element_text(size=14),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size=14),
        axis.text.y = element_text(size = 12),
        legend.title = element_text(size=14),
        legend.text = element_text(size=12)) +
  scale_fill_brewer(palette = "Dark2", guide="none") 
comprehension_bar_graph
ggsave(comprehension_bar_graph, file="../graphs/comprehension_nonIC_sent_why_period_mean_all_models.pdf", width=8, height=4)

# 3. Statistical analysis ----
## Llama3.2-1B ----
comprehension.llama1B <- comprehension.llama1B.data %>%
  select(c("item_id", "verb", "verb_type","exp","nonexp")) %>%
  pivot_longer(cols=c(exp, nonexp),
               names_to = "rc_type",
               values_to = "critical_region_logprob") %>%
  mutate(rc_type=fct_relevel(rc_type,ref="nonexp"),
         verb_type=fct_relevel(verb_type, ref="IC")) %>%
  na.omit() # %>%
#   filter(verb_type=="IC")
# llama1B_analysis <- lmer(critical_region_logprob ~ rc_type + (1|item_id),
#                          comprehension.llama1B)
# summary(llama1B_analysis)
llama1B_analysis <- lmer(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                         comprehension.llama1B)
summary(llama1B_analysis)

## Llama3.2-1B-Instruct ----
comprehension.llama1B.instruct <- comprehension.llama1B.instruct.data %>%
  select(c("item_id", "verb", "verb_type","exp","nonexp")) %>%
  pivot_longer(cols=c(exp, nonexp),
               names_to = "rc_type",
               values_to = "critical_region_logprob") %>%
  mutate(rc_type=fct_relevel(rc_type,ref="nonexp"),
         verb_type=fct_relevel(verb_type, ref="IC")) %>%
  na.omit()
llama1B_instruct_analysis <- lmer(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                  comprehension.llama1B.instruct)
summary(llama1B_instruct_analysis)

## Llama3.2-3B ----
comprehension.llama3B <- comprehension.llama3B.data %>%
  select(c("item_id", "verb", "verb_type","exp","nonexp")) %>%
  pivot_longer(cols=c(exp, nonexp),
               names_to = "rc_type",
               values_to = "critical_region_logprob") %>%
  mutate(rc_type=fct_relevel(rc_type,ref="nonexp"),
         verb_type=fct_relevel(verb_type, ref="IC")) # %>%
#   filter(verb_type=="IC")
# llama3B_analysis <- lmer(critical_region_logprob ~ continuation_type + (1|item_id),
#                          comprehension.llama3B)
# summary(llama3B_analysis)
llama3B_analysis <- lmer(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                         comprehension.llama3B)
summary(llama3B_analysis)

## Llama3.2-3B-Instruct ----
comprehension.llama3B.instruct <- comprehension.llama3B.instruct.data %>%
  select(c("item_id", "verb", "verb_type","exp","nonexp")) %>%
  pivot_longer(cols=c(exp, nonexp),
               names_to = "rc_type",
               values_to = "critical_region_logprob") %>%
  mutate(rc_type=fct_relevel(rc_type,ref="nonexp"),
         verb_type=fct_relevel(verb_type, ref="IC"))

llama3B_instruct_analysis <- lmer(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                         comprehension.llama3B.instruct)
summary(llama3B_instruct_analysis)

## gpt2 ----
comprehension.gpt2 <- comprehension.gpt2.data %>%
  select(c("item_id", "verb", "verb_type","exp","nonexp")) %>%
  pivot_longer(cols=c(exp, nonexp),
               names_to = "continuation_type",
               values_to = "critical_region_logprob") %>%
  mutate(critical_region_logprob=critical_region_logprob*8) %>% 
  mutate(continuation_type=fct_relevel(continuation_type, ref="nonexp"),
         verb_type=fct_relevel(verb_type, ref="IC")) # %>%
  # filter(verb_type=="IC")
# contrasts(comprehension.gpt2$continuation_type)=contr.sum(2)
# levels(comprehension.gpt2$continuation_type)
# gpt2_analysis <- lmer(critical_region_logprob ~ continuation_type + (1|item_id),
#                          comprehension.gpt2)
# summary(gpt2_analysis)
gpt2_analysis <- lmer(critical_region_logprob ~ continuation_type * verb_type + (1+continuation_type+ verb_type|item_id),
                         comprehension.gpt2)
summary(gpt2_analysis)

# -----------------------------------------------------------------------------------

