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
library(grid)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")


# 1. Data ----
## Llama3.2-1B ----
comprehension.llama1B.because.data <- read.csv("../../../data/comprehension_nonIC_sent_because/comprehension_nonIC_sent_because_Llama-3.2-1B_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_logprob)

comprehension.llama1B.and.so.data <- read.csv("../../../data/comprehension_nonIC_sent_and_so/comprehension_nonIC_sent_and_so_Llama-3.2-1B_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_sum) %>% 
  rename(critical_region_logprob = "critical_region_sum")

comprehension.llama1B.data <- bind_rows(lst(comprehension.llama1B.because.data, comprehension.llama1B.and.so.data), .id="connective") %>% 
  rename(rc_type = "continuation_type") %>% 
  mutate(connective=if_else(connective=="comprehension.llama1B.because.data", "because", "and_so"))

comprehension.llama1B.logodds <- comprehension.llama1B.data %>% 
  pivot_wider(names_from = connective, values_from = critical_region_logprob) %>%
  mutate(logodds = because-and_so,
         surp_because = -because,
         surp_and_so = -and_so,
         surp_diff = surp_because - surp_and_so)

llama1B_logodds_mean <- comprehension.llama1B.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

llama1B_logodds_surp_mean <- comprehension.llama1B.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(surp_diff),
            CILow = ci.low(surp_diff),
            CIHigh = ci.high(surp_diff)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

## Llama3.2-1B-Instruct ----
comprehension.llama1B_instruct.because.data <- read.csv("../../../data/comprehension_nonIC_sent_because/comprehension_nonIC_sent_because_Llama-3.2-1B-Instruct_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_logprob)

comprehension.llama1B_instruct.and.so.data <- read.csv("../../../data/comprehension_nonIC_sent_and_so/comprehension_nonIC_sent_and_so_Llama-3.2-1B-Instruct_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_sum) %>% 
  rename(critical_region_logprob = "critical_region_sum")

comprehension.llama1B_instruct.data <- bind_rows(lst(comprehension.llama1B_instruct.because.data, comprehension.llama1B_instruct.and.so.data), .id="connective") %>% 
  rename(rc_type = "continuation_type") %>% 
  mutate(connective=if_else(connective=="comprehension.llama1B_instruct.because.data", "because", "and_so"))

comprehension.llama1B_instruct.logodds <- comprehension.llama1B_instruct.data %>% 
  pivot_wider(names_from = connective, values_from = critical_region_logprob) %>%
  mutate(logodds = because-and_so,
         surp_because = -because,
         surp_and_so = -and_so,
         surp_diff = surp_because - surp_and_so)

llama1B_instruct_logodds_mean <- comprehension.llama1B_instruct.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

llama1B_instruct_surp_mean <- comprehension.llama1B_instruct.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(surp_diff),
            CILow = ci.low(surp_diff),
            CIHigh = ci.high(surp_diff)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

## Llama3.2-3B ----
comprehension.llama3B.because.data <- read.csv("../../../data/comprehension_nonIC_sent_because/comprehension_nonIC_sent_because_Llama-3.2-3B_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_logprob)

comprehension.llama3B.and.so.data <- read.csv("../../../data/comprehension_nonIC_sent_and_so/comprehension_nonIC_sent_and_so_Llama-3.2-3B_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_sum) %>% 
  rename(critical_region_logprob = "critical_region_sum")

comprehension.llama3B.data <- bind_rows(lst(comprehension.llama3B.because.data, comprehension.llama3B.and.so.data), .id="connective") %>% 
  rename(rc_type = "continuation_type") %>% 
  mutate(connective=if_else(connective=="comprehension.llama3B.because.data", "because", "and_so"))

comprehension.llama3B.logodds <- comprehension.llama3B.data %>% 
  pivot_wider(names_from = connective, values_from = critical_region_logprob) %>%
  mutate(logodds = because-and_so,
         surp_because = -because,
         surp_and_so = -and_so,
         surp_diff = surp_because - surp_and_so)

llama3B_logodds_mean <- comprehension.llama3B.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

llama3B_surp_mean <- comprehension.llama3B.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(surp_diff),
            CILow = ci.low(surp_diff),
            CIHigh = ci.high(surp_diff)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

## Llama3.2-3B-Instruct ----
comprehension.llama3B_instruct.because.data <- read.csv("../../../data/comprehension_nonIC_sent_because/comprehension_nonIC_sent_because_Llama-3.2-3B-Instruct_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_logprob)

comprehension.llama3B_instruct.and.so.data <- read.csv("../../../data/comprehension_nonIC_sent_and_so/comprehension_nonIC_sent_and_so_Llama-3.2-3B-Instruct_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_sum) %>% 
  rename(critical_region_logprob = "critical_region_sum")

comprehension.llama3B_instruct.data <- bind_rows(lst(comprehension.llama3B_instruct.because.data, comprehension.llama3B_instruct.and.so.data), .id="connective") %>% 
  rename(rc_type = "continuation_type") %>% 
  mutate(connective=if_else(connective=="comprehension.llama3B_instruct.because.data", "because", "and_so"))

comprehension.llama3B_instruct.logodds <- comprehension.llama3B_instruct.data %>% 
  pivot_wider(names_from = connective, values_from = critical_region_logprob) %>%
  mutate(logodds = because-and_so,
         surp_because = -because,
         surp_and_so = -and_so,
         surp_diff = surp_because - surp_and_so)

llama3B_instruct_logodds_mean <- comprehension.llama3B_instruct.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

llama3B_surp_logodds_mean <- comprehension.llama3B_instruct.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(surp_diff),
            CILow = ci.low(surp_diff),
            CIHigh = ci.high(surp_diff)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)


## gpt2 ----
comprehension.gpt2.because.data <- read.csv("../../../data/comprehension_nonIC_sent_because/comprehension_nonIC_sent_because_gpt2_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_logprob)

comprehension.gpt2.and.so.data <- read.csv("../../../data/comprehension_nonIC_sent_and_so/comprehension_nonIC_sent_and_so_gpt2_1.csv", header=TRUE) %>%
  na.omit() %>% 
  select(sent_id, item_id, verb, verb_type, continuation_type, critical_region_sum) %>% 
  rename(critical_region_logprob = "critical_region_sum")

comprehension.gpt2.data <- bind_rows(lst(comprehension.gpt2.because.data, comprehension.gpt2.and.so.data), .id="connective") %>% 
  rename(rc_type = "continuation_type") %>% 
  mutate(connective=if_else(connective=="comprehension.gpt2.because.data", "because", "and_so"))

comprehension.gpt2.logodds <- comprehension.gpt2.data %>% 
  pivot_wider(names_from = connective, values_from = critical_region_logprob) %>%
  mutate(logodds = because-and_so,
         surp_because = -because,
         surp_and_so = -and_so,
         surp_diff = surp_because - surp_and_so)

gpt2_logodds_mean <- comprehension.gpt2.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

gpt2_surp_mean <- comprehension.gpt2.logodds %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(surp_diff),
            CILow = ci.low(surp_diff),
            CIHigh = ci.high(surp_diff)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)


## all models ----
comprehension.data <- bind_rows(lst(comprehension.llama1B.data,comprehension.llama1B_instruct.data,comprehension.llama3B.data, comprehension.llama3B_instruct.data, comprehension.gpt2.data), .id="model") %>% 
  mutate(model=case_when(model == "comprehension.llama1B.data" ~ "1B",
                         model == "comprehension.llama1B_instruct.data" ~ "1B-Instruct",
                         model == "comprehension.llama3B.data" ~ "3B",
                         model == "comprehension.llama3B_instruct.data" ~ "3B-Instruct",
                         model == "comprehension.gpt2.data" ~ "GPT2")) %>% 
  mutate(critical_region_surp = -critical_region_logprob)

comprehension.because.data <- comprehension.data %>% 
  filter(connective == "because")

comprehension.and.so.data <- comprehension.data %>% 
  filter(connective == "and_so")

comprehension.logodds <- bind_rows(lst(comprehension.llama1B.logodds,comprehension.llama1B_instruct.logodds, comprehension.llama3B.logodds, comprehension.llama3B_instruct.logodds, comprehension.gpt2.logodds), .id="model") %>% 
  mutate(model=case_when(model == "comprehension.llama1B.logodds" ~ "1B",
                         model == "comprehension.llama1B_instruct.logodds" ~ "1B-Instruct",
                         model == "comprehension.llama3B.logodds" ~ "3B",
                         model == "comprehension.llama3B_instruct.logodds" ~ "3B-Instruct",
                         model == "comprehension.gpt2.logodds" ~ "GPT2"))

## all models (mean) ----
comprehension_mean <- comprehension.data %>% 
  group_by(model,verb_type, rc_type, connective) %>% 
  summarize(Mean = mean(critical_region_logprob),
            CILow = ci.low(critical_region_logprob),
            CIHigh = ci.high(critical_region_logprob)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

comprehension_IC_mean <- comprehension_mean %>% 
  filter(verb_type == "IC")

comprehension_nonIC_mean <- comprehension_mean %>% 
  filter(verb_type == "nonIC")

comprehension_surp_mean <- comprehension.data %>% 
  group_by(model,verb_type, rc_type, connective) %>% 
  summarize(Mean = mean(critical_region_surp),
            CILow = ci.low(critical_region_surp),
            CIHigh = ci.high(critical_region_surp)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

comprehension_surp_IC_mean <- comprehension_surp_mean %>% 
  filter(verb_type == "IC")

comprehension_surp_nonIC_mean <- comprehension_surp_mean %>% 
  filter(verb_type == "nonIC")

comprehension_surp_diff_mean <- comprehension.data %>% 
  select(-c(critical_region_logprob, sent_id)) %>% 
  pivot_wider(names_from = rc_type, values_from = critical_region_surp) %>% 
  mutate(critical_region_diff = exp-nonexp) %>% 
  group_by(model,verb_type, connective) %>% 
  summarize(Mean = mean(critical_region_diff),
            CILow = ci.low(critical_region_diff),
            CIHigh = ci.high(critical_region_diff)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

comprehension_surp_diff_IC_mean <- comprehension_surp_diff_mean %>% 
  filter(verb_type == "IC")

comprehension_surp_diff_nonIC_mean <- comprehension_surp_diff_mean %>% 
  filter(verb_type == "nonIC")

comprehension_logodds_mean <- comprehension.logodds %>% 
  group_by(model, verb_type, rc_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

comprehension_surp_mean <- comprehension.logodds %>% 
  group_by(model, verb_type, rc_type) %>% 
  summarize(Mean = mean(surp_diff),
            CILow = ci.low(surp_diff),
            CIHigh = ci.high(surp_diff)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

# 2. Plot ----
#### density of log odds ----
comprehension_graph <- ggplot(comprehension.logodds,
                              aes(x=logodds,
                                  fill=verb_type,
                                  pattern=rc_type)) +
  geom_density_pattern(
    pattern_angle = 45,
    pattern_spacing = 0.02,
    pattern_fill="black",
    pattern_alpha=0.6,
    alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "RC type") +
  coord_cartesian(clip="off") +
  facet_grid(model ~ verb_type) +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10)) +
  labs(y = "density",
       x = "log odds (because - and so)")
comprehension_graph
ggsave(comprehension_graph, file="../graphs/comprehension_because_llama.pdf", width=8, height=4)

### line graph for IC verbs mean ----
comprehension_ic_line_graph <- ggplot(comprehension_mean %>% 
                                       filter(verb_type == "IC"),
                                     aes(x=rc_type,y=Mean,
                                         group = connective)) +
  geom_point(aes(shape = connective),
    stat="identity",
    alpha=0.7,
    size=2) +
  geom_line(aes(linetype = connective)) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  facet_grid(. ~ model) + 
  labs(y = "Mean log probabilities",
       x = "RC type") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 10),
        axis.title.y = element_text(size = 14))
comprehension_ic_line_graph
ggsave(comprehension_ic_line_graph, file="../graphs/comprehension_ic_sent_line_graph.pdf", width=8, height=4)

### line graph for IC verbs surp mean ----
comprehension_surp_ic_line_graph <- ggplot(comprehension_surp_IC_mean %>% 
                                             mutate(connective = if_else(connective == "and_so", "and so", "because")),
                                      aes(x=rc_type,y=Mean,
                                          group = connective)) +
  geom_point(aes(shape = connective),
             stat="identity",
             alpha=0.7,
             size=2) +
  geom_line(aes(linetype = connective)) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  facet_grid(. ~ model) + 
  labs(y = "Mean surprisal",
       x = "RC type") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14),
        axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 10),
        axis.title.y = element_text(size = 14))
comprehension_surp_ic_line_graph
ggsave(comprehension_surp_ic_line_graph, file="../graphs/comprehension_ic_surp_line_graph.pdf", width=8, height=4)


comprehension_surp_ic_diff_line_graph <- ggplot(comprehension_surp_diff_IC_mean %>% 
                                             mutate(connective = if_else(connective == "and_so", "and so", "because")),
                                           aes(x=connective,y=Mean)) +
  geom_point(aes(shape = connective),
             stat="identity",
             alpha=0.7,
             size=2,
             show.legend=FALSE) +
  geom_line(aes(group=1),linetype="dotted") +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  facet_grid(. ~ model) + 
  labs(y = "Mean surprisal difference of\n the connective (exp- nonexp)",
       x = "Connective") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        legend.text = element_text(size=12),
        legend.title = element_text(size=14),
        axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 10),
        axis.title.y = element_text(size = 14))
comprehension_surp_ic_diff_line_graph
ggsave(comprehension_surp_ic_diff_line_graph, file="../graphs/comprehension_ic_surp_diff_line_graph.pdf", width=8, height=4)

### bar graph for logodds mean ----
comprehension_bar_graph <- ggplot(comprehension_logodds_mean,
                                   aes(x=verb_type, y=Mean,
                                       fill=verb_type,
                                       pattern=rc_type)) +
  geom_bar_pattern(
    position = "dodge",
    stat="identity",
    pattern_angle = 45,
    pattern_spacing = 0.02,
    pattern_fill="black",
    pattern_alpha=0.6,
    alpha=0.7) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                position=position_dodge(width=0.8),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2", name="Verb Type") +
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  facet_wrap(. ~ model) + 
  labs(y = "Mean log odds (because - and so)",
       x = "Verb type") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10))
comprehension_bar_graph
ggsave(comprehension_bar_graph, file="../graphs/comprehension_bar_and_so_llama.pdf", width=7, height=4)

### bar graph for surp mean ----
comprehension_bar_graph <- ggplot(comprehension_surp_mean,
                                  aes(x=verb_type, y=Mean,
                                      fill=verb_type,
                                      pattern=rc_type)) +
  geom_bar_pattern(
    position = "dodge",
    stat="identity",
    pattern_angle = 45,
    pattern_spacing = 0.02,
    pattern_fill="black",
    pattern_alpha=0.6,
    alpha=0.7) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                position=position_dodge(width=0.8),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2", name="Verb Type") +
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  facet_wrap(. ~ model) + 
  labs(y = "Mean surprisal (because - and so)",
       x = "Verb type") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10))
comprehension_bar_graph

## bar graph for IC ----
comprehension_ic_bar_graph <- ggplot(comprehension_IC_mean,
                                  aes(x=connective, y=Mean,
                                      fill=connective,
                                      pattern=rc_type)) +
  geom_bar_pattern(
    position = "dodge",
    stat="identity",
    pattern_angle = 45,
    pattern_spacing = 0.02,
    pattern_fill="black",
    pattern_alpha=0.6,
    alpha=0.7) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                position=position_dodge(width=0.8),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  scale_fill_manual(values = cbPalette, 
                    name="Connective") +
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  facet_wrap(. ~ model) + 
  labs(y = "Mean log prob of connectives",
       x = "Verb type") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10))
comprehension_ic_bar_graph

# 3. Statistical analysis ----
## log-odds ----
### gpt2 ----
# summary(gpt2_analysis)
comprehension.gpt2.logodds <- comprehension.gpt2.logodds %>% 
  filter(verb_type == "IC") %>% 
  mutate(rc_type = fct_relevel(rc_type, ref="exp"))
# contrasts(comprehension.gpt2.logodds$rc_type)=contr.sum(2)
# contrasts(comprehension.gpt2.logodds$verb_type)=contr.sum(2)
gpt2_analysis <- lmer(logodds ~ rc_type + (1|item_id),
                      comprehension.gpt2.logodds)
summary(gpt2_analysis)

### Llama3.2-1B ----
comprehension.llama1B.logodds <- comprehension.llama1B.logodds %>% 
  filter(verb_type == "IC") %>% 
  mutate(rc_type = fct_relevel(rc_type,ref="exp"))
# contrasts(comprehension.llama1B.logodds$rc_type)=contr.sum(2)
# contrasts(comprehension.llama1B.logodds$verb_type)=contr.sum(2)
llama1B_analysis <- lmer(logodds ~ rc_type + (1|item_id),
                         comprehension.llama1B.logodds)
summary(llama1B_analysis)

### Llama3.2-3B ----
comprehension.llama3B.logodds <- comprehension.llama3B.logodds %>% 
  filter(verb_type == "IC") %>% 
  mutate(rc_type = fct_relevel(rc_type,ref="exp"))
# contrasts(comprehension.llama3B.logodds$rc_type)=contr.sum(2)
# contrasts(comprehension.llama3B.logodds$verb_type)=contr.sum(2)
llama3B_analysis <- lmer(logodds ~ rc_type + (1|item_id),
                         comprehension.llama3B.logodds)
summary(llama3B_analysis)

### Llama3.2-1B-Instruct ----
comprehension.llama1B_instruct.logodds <- comprehension.llama1B_instruct.logodds %>% 
  filter(verb_type == "IC") %>% 
  mutate(rc_type = fct_relevel(rc_type,ref="exp"))
llama1B_instruct_analysis <- lmer(logodds ~ rc_type + (1|item_id),
                                  comprehension.llama1B_instruct.logodds)
summary(llama1B_instruct_analysis)

### Llama3.2-3B-Instruct ----
comprehension.llama3B_instruct.logodds <- comprehension.llama3B_instruct.logodds %>% 
  filter(verb_type == "IC") %>% 
  mutate(rc_type = fct_relevel(rc_type,ref="exp"))
llama3B_instruct_analysis <- lmer(logodds ~ rc_type + (1|item_id),
                         comprehension.llama3B_instruct.logodds)
summary(llama3B_instruct_analysis)

## IC verbs, because vs. and_so ----
### gpt2 ----
comprehension.gpt2.ic <- comprehension.gpt2.data %>%
  filter(verb_type == "IC") %>% 
  mutate(critical_region_surp = -critical_region_logprob,
         rc_type = fct_relevel(rc_type,"exp"),
         connective = fct_relevel(connective,"because"))
# contrasts(comprehension.gpt2.ic$rc_type)=contr.sum(2)
# contrasts(comprehension.gpt2.ic$connective)=contr.sum(2)

# using surprisal 
gpt2_connective_surp_analysis <- lmer(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                 comprehension.gpt2.ic)
summary(gpt2_connective_surp_analysis)

gpt2_connective_surp_bayesian_default_prior <- brm(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                              data=comprehension.gpt2.ic,
                                              iter=8000,
                                              warmup = 4000,
                                              chains=4,
                                              cores=4,
                                              control=list(max_treedepth = 15, adapt_delta = 0.99),
                                              file="../cache/gpt2_connective_surp_bayesian_default_prior",
                                              seed=1024)
summary(gpt2_connective_surp_bayesian_default_prior)
emmeans(gpt2_connective_surp_bayesian_default_prior, pairwise ~ rc_type | connective ,adjust = "bonferroni")

# using log odds 
gpt2_connective_analysis <- lmer(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                 comprehension.gpt2.ic)
summary(gpt2_connective_analysis)

gpt2_connective_bayesian_default_prior <- brm(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                      data=comprehension.gpt2.ic,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/gpt2_connective_bayesian_default_prior",
                                      seed=1024)
summary(gpt2_connective_bayesian_default_prior)

### Llama3.2-1B ----
comprehension.llama1B.ic <- comprehension.llama1B.data %>% 
  filter(verb_type == "IC") %>% 
  mutate(critical_region_surp = -critical_region_logprob,
         rc_type = fct_relevel(rc_type,"exp"),
         connective = fct_relevel(connective,"because"))
# contrasts(comprehension.llama1B.ic$rc_type)=contr.sum(2)
# contrasts(comprehension.llama1B.ic$connective)=contr.sum(2)

# using surprisal
llama1B_connective_surp_analysis <- lmer(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                    comprehension.llama1B.ic)
summary(llama1B_connective_surp_analysis)

llama1B_connective_surp_bayesian_default_prior <- brm(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                                 data=comprehension.llama1B.ic,
                                                 iter=8000,
                                                 warmup = 4000,
                                                 chains=4,
                                                 cores=4,
                                                 control=list(max_treedepth = 15, adapt_delta = 0.99),
                                                 file="../cache/llama1B_connective_surp_bayesian_default_prior",
                                                 seed=1024)
summary(llama1B_connective_surp_bayesian_default_prior)
emmeans(llama1B_connective_surp_bayesian_default_prior, pairwise ~ rc_type | connective ,adjust = "bonferroni")

# using log odds
llama1B_connective_analysis <- lmer(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                 comprehension.llama1B.ic)
summary(llama1B_connective_analysis)

llama1B_connective_bayesian_default_prior <- brm(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                              data=comprehension.llama1B.ic,
                                              iter=8000,
                                              warmup = 4000,
                                              chains=4,
                                              cores=4,
                                              control=list(max_treedepth = 15, adapt_delta = 0.99),
                                              file="../cache/llama1B_connective_bayesian_default_prior",
                                              seed=1024)
summary(llama1B_connective_bayesian_default_prior)

### Llama3.2-3B ----
comprehension.llama3B.ic <- comprehension.llama3B.data %>% 
  filter(verb_type == "IC") %>% 
  mutate(critical_region_surp = -critical_region_logprob,
         rc_type = fct_relevel(rc_type,"exp"),
         connective = fct_relevel(connective, "because"))
# contrasts(comprehension.llama3B.ic$rc_type)=contr.sum(2)
# contrasts(comprehension.llama3B.ic$connective)=contr.sum(2)

# using surprisal
llama3B_connective_surp_analysis <- lmer(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                    comprehension.llama3B.ic)
summary(llama3B_connective_surp_analysis)

llama3B_connective_surp_bayesian_default_prior <- brm(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                                 data=comprehension.llama3B.ic,
                                                 iter=8000,
                                                 warmup = 4000,
                                                 chains=4,
                                                 cores=4,
                                                 control=list(max_treedepth = 15, adapt_delta = 0.99),
                                                 file="../cache/llama3B_connective_surp_bayesian_default_prior",
                                                 seed=1024)
summary(llama3B_connective_surp_bayesian_default_prior)
emmeans(llama3B_connective_surp_bayesian_default_prior, pairwise ~ rc_type | connective ,adjust = "bonferroni")

# using log probs
llama3B_connective_analysis <- lmer(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                    comprehension.llama3B.ic)
summary(llama3B_connective_analysis)

llama3B_connective_bayesian_default_prior <- brm(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                                 data=comprehension.llama3B.ic,
                                                 iter=8000,
                                                 warmup = 4000,
                                                 chains=4,
                                                 cores=4,
                                                 control=list(max_treedepth = 15, adapt_delta = 0.99),
                                                 file="../cache/llama3B_connective_bayesian_default_prior",
                                                 seed=1024)
summary(llama3B_connective_bayesian_default_prior)

### Llama3.2-1B-Instruct ----
comprehension.llama1B_instruct.ic <- comprehension.llama1B_instruct.data %>% 
  filter(verb_type == "IC") %>% 
  mutate(critical_region_surp = -critical_region_logprob,
         rc_type = fct_relevel(rc_type,"exp"),
         connective = fct_relevel(connective,"because"))
# contrasts(comprehension.llama1B_instruct.ic$rc_type)=contr.sum(2)
# contrasts(comprehension.llama1B_instruct.ic$connective)=contr.sum(2)
# using surprisal 
llama1B_instruct_connective_surp_analysis <- lmer(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                             comprehension.llama1B_instruct.ic)
summary(llama1B_instruct_connective_surp_analysis)

llama1B_instruct_connective_surp_bayesian_default_prior <- brm(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                                          data=comprehension.llama1B_instruct.ic,
                                                          iter=8000,
                                                          warmup = 4000,
                                                          chains=4,
                                                          cores=4,
                                                          control=list(max_treedepth = 15, adapt_delta = 0.99),
                                                          file="../cache/llama1B_instruct_connective_surp_bayesian_default_prior",
                                                          seed=1024)
summary(llama1B_instruct_connective_surp_bayesian_default_prior)
emmeans(llama1B_instruct_connective_surp_bayesian_default_prior, pairwise ~ rc_type | connective ,adjust = "bonferroni")

# using log probs
llama1B_instruct_connective_analysis <- lmer(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                    comprehension.llama1B_instruct.ic)
summary(llama1B_instruct_connective_analysis)

llama1B_instruct_connective_bayesian_default_prior <- brm(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                                 data=comprehension.llama1B_instruct.ic,
                                                 iter=8000,
                                                 warmup = 4000,
                                                 chains=4,
                                                 cores=4,
                                                 control=list(max_treedepth = 15, adapt_delta = 0.99),
                                                 file="../cache/llama1B_instruct_connective_bayesian_default_prior",
                                                 seed=1024)
summary(llama1B_instruct_connective_bayesian_default_prior)

### Llama3.2-3B-Instruct ----
comprehension.llama3B_instruct.ic <- comprehension.llama3B_instruct.data %>% 
  filter(verb_type == "IC") %>% 
  mutate(critical_region_surp=-critical_region_logprob,
         rc_type = fct_relevel(rc_type,"exp"),
         connective = fct_relevel(connective,"because"))
# contrasts(comprehension.llama3B_instruct.ic$rc_type)=contr.sum(2)
# contrasts(comprehension.llama3B_instruct.ic$connective)=contr.sum(2)
# using surprisal
llama3B_instruct_connective_surp_analysis <- lmer(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                             comprehension.llama3B_instruct.ic)
summary(llama3B_instruct_connective_surp_analysis)

llama3B_instruct_connective_surp_bayesian_default_prior <- brm(critical_region_surp ~ connective * rc_type + (1+rc_type+connective|item_id),
                                                          data=comprehension.llama3B_instruct.ic,
                                                          iter=8000,
                                                          warmup = 4000,
                                                          chains=4,
                                                          cores=4,
                                                          control=list(max_treedepth = 15, adapt_delta = 0.99),
                                                          file="../cache/llama3B_instruct_connective_surp_bayesian_default_prior",
                                                          seed=1024)
summary(llama3B_instruct_connective_surp_bayesian_default_prior)
emmeans(llama3B_instruct_connective_surp_bayesian_default_prior, pairwise ~ rc_type | connective ,adjust = "bonferroni")

# using log probs
llama3B_instruct_connective_analysis <- lmer(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                             comprehension.llama3B_instruct.ic)
summary(llama3B_instruct_connective_analysis)

llama3B_instruct_connective_bayesian_default_prior <- brm(critical_region_logprob ~ connective * rc_type + (1+rc_type+connective|item_id),
                                                          data=comprehension.llama3B_instruct.ic,
                                                          iter=8000,
                                                          warmup = 4000,
                                                          chains=4,
                                                          cores=4,
                                                          control=list(max_treedepth = 15, adapt_delta = 0.99),
                                                          file="../cache/llama3B_instruct_connective_bayesian_default_prior",
                                                          seed=1024)
summary(llama3B_instruct_connective_bayesian_default_prior)

