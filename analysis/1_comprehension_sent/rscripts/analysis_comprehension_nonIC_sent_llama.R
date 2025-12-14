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


# 1. Data (Simple question) ----
## Llama3.2-1B ----
comprehension.llama1B.data <- read.csv("../../../data/comprehension_nonIC_full_alt_llama/comprehension_nonIC_full_alt_llama_Llama-3.2-1B_1.csv", header=TRUE) %>%
  na.omit()
comprehension.llama1B.data <- comprehension.llama1B.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  rename(rc_type = "continuation_type") %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

comprehension.llama1B.binary <- comprehension.llama1B.data %>% 
  mutate(response = if_else(Yes > No, "Yes", "No"))

## Llama3.2-1B-Instruct ----
comprehension.llama1B.instruct.data <- read.csv("../../../data/comprehension_nonIC_full_alt_llama/comprehension_nonIC_full_alt_llama_Llama-3.2-1B-Instruct_1.csv", header=TRUE) %>%
  na.omit()
comprehension.llama1B.instruct.data <- comprehension.llama1B.instruct.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  rename(rc_type = "continuation_type") %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

comprehension.llama1B.instruct.binary <- comprehension.llama1B.instruct.data %>% 
  mutate(response = if_else(Yes > No, "Yes", "No"))

## Llama3.2-3B ----
comprehension.llama3B.data <- read.csv("../../../data/comprehension_nonIC_full_alt_llama/comprehension_nonIC_full_alt_llama_Llama-3.2-3B_1.csv", header=TRUE) %>%
  na.omit()
comprehension.llama3B.data <- comprehension.llama3B.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  rename(rc_type = "continuation_type") %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

comprehension.llama3B.binary <- comprehension.llama3B.data %>% 
  mutate(response = if_else(Yes > No, "Yes", "No"))

## Llama3.2-3B-Instruct ----
comprehension.llama3B.instruct.data <- read.csv("../../../data/comprehension_nonIC_full_alt_llama/comprehension_nonIC_full_alt_llama_Llama-3.2-3B-Instruct_1.csv", header=TRUE) %>%
  na.omit()
comprehension.llama3B.instruct.data <- comprehension.llama3B.instruct.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  rename(rc_type = "continuation_type") %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

comprehension.llama3B.instruct.binary <- comprehension.llama3B.instruct.data %>% 
  mutate(response = if_else(Yes > No, "Yes", "No"))

## GPT-2 ----
comprehension.gpt2.data <- read.csv("../../../data/comprehension_nonIC_full_alt_llama/comprehension_nonIC_full_alt_llama_gpt2_1.csv", header=TRUE) %>%
  na.omit()
comprehension.gpt2.data <- comprehension.gpt2.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  rename(rc_type = "continuation_type") %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

comprehension.gpt2.binary <- comprehension.gpt2.data %>% 
  mutate(response = if_else(Yes > No, "Yes", "No"))

## all models ----
comprehension.data <- bind_rows(lst(comprehension.llama1B.data,comprehension.llama1B.instruct.data,comprehension.llama3B.data,comprehension.llama3B.instruct.data,comprehension.gpt2.data), .id="model") %>% 
  mutate(model=case_when(model == "comprehension.llama1B.data" ~ "Llama3.2-1B",
                         model == "comprehension.llama3B.data" ~ "Llama3.2-3B",
                         model == "comprehension.llama1B.instruct.data" ~ "Llama3.2-1B-Instruct",
                         model == "comprehension.llama3B.instruct.data" ~ "Llama3.2-3B-Instruct",
                         model == "comprehension.gpt2.data" ~ "GPT2"))

## all models (mean) ----
comprehension_mean <- comprehension.data %>% 
  select(-c(Yes, No)) %>% 
  group_by(model,verb_type, rc_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

# 2. Plot ----
## combined ----
### density of log odds ----
comprehension_graph <- ggplot(comprehension.data,
                              aes(x=logodds,
                                  fill=verb_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  # annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
  #                                                length=unit(2,"mm")),
  #                                    gp=gpar(col="black", lwd=1.5)), xmin = -1.1, xmax = 1.7, ymin = -0.62, ymax = -0.62) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer Yes", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = 1.5, xmax = 2.0, ymin = -0.68, ymax = -0.68) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer No", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -1.9, xmax = -0.9, ymin = -0.68, ymax = -0.68) +
  # coord_cartesian(clip="off") +
  facet_grid(model ~ .) +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size=12),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size=12)) +
  labs(y = "density",
       x = "log odds (yes vs. no)")
comprehension_graph
ggsave(comprehension_graph, file="../graphs/comprehension_nonIC_sent_llama.pdf", width=8, height=4)


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
  labs(y = "Mean log probability difference (Yes - No)",
       x = "Verb Type") +
  facet_grid(. ~ model) +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10)) +
  scale_fill_brewer(palette = "Dark2",guide="none") 
comprehension_bar_graph
ggsave(comprehension_bar_graph, file="../graphs/comprehension_nonIC_sent_llama_mean.pdf", width=7, height=4)

# 3. Statistical analysis ----
## Llama3.2-1B ----
comprehension.llama1B.data <- comprehension.llama1B.data %>%
  mutate(verb_type = fct_relevel(verb_type, ref="IC"),
         continuation_type = fct_relevel(rc_type, ref="exp"))
comprehension.llama1B.data <- comprehension.llama1B.data %>%
  mutate(verb_type = fct_relevel(verb_type),
         rc_type = fct_relevel(rc_type))
contrasts(comprehension.llama1B.data$verb_type)=contr.sum(2)
contrasts(comprehension.llama1B.data$rc_type)=contr.sum(2)

llama1B_analysis <- lmer(logodds ~ verb_type * continuation_type + 
                           (1+verb_type+rc_type|item_id),
                         comprehension.llama1B.data)
summary(llama1B_analysis)

## Llama3.2-1B-Instruct ----
comprehension.llama1B.instruct.data <- comprehension.llama1B.instruct.data %>%
  mutate(verb_type = fct_relevel(verb_type),
         rc_type = fct_relevel(rc_type))
contrasts(comprehension.llama1B.instruct.data$verb_type)=contr.sum(2)
contrasts(comprehension.llama1B.instruct.data$rc_type)=contr.sum(2)

llama1B_instruct_analysis <- lmer(logodds ~ verb_type * rc_type + 
                           (1+verb_type+rc_type|item_id),
                         comprehension.llama1B.instruct.data)
summary(llama1B_instruct_analysis)

## Llama3.2-3B ----
comprehension.llama3B.data <- comprehension.llama3B.data %>%
  mutate(verb_type = fct_relevel(verb_type, ref="IC"),
         rc_type = fct_relevel(rc_type, ref="exp"))
comprehension.llama3B.data <- comprehension.llama3B.data %>%
  mutate(verb_type = fct_relevel(verb_type),
         rc_type = fct_relevel(crc_type))
contrasts(comprehension.llama3B.data$verb_type)=contr.sum(2)
contrasts(comprehension.llama3B.data$rc_type)=contr.sum(2)

llama3B_analysis <- lmer(logodds ~ verb_type * rc_type + 
                           (1+verb_type+rc_type|item_id),
                         comprehension.llama3B.data)
summary(llama3B_analysis)

## Llama3.2-3B-Instruct ----
comprehension.llama3B.instruct.data <- comprehension.llama3B.instruct.data %>%
  mutate(verb_type = fct_relevel(verb_type,ref="IC"),
         rc_type = fct_relevel(rc_type,ref="exp"))
comprehension.llama3B.instruct.data <- comprehension.llama3B.instruct.data %>%
  mutate(verb_type = fct_relevel(verb_type),
         rc_type = fct_relevel(rc_type))
contrasts(comprehension.llama3B.instruct.data$verb_type)=contr.sum(2)
contrasts(comprehension.llama3B.instruct.data$rc_type)=contr.sum(2)

llama3B_instruct_analysis <- lmer(logodds ~ verb_type * rc_type + 
                                    (1+verb_type+rc_type|item_id),
                                  comprehension.llama3B.instruct.data)
summary(llama3B_instruct_analysis)

## GPT-2 ----
comprehension.gpt2.data <- comprehension.gpt2.data %>%
  mutate(verb_type = fct_relevel(verb_type, ref="IC"),
         rc_type = fct_relevel(rc_type, ref="exp"))
comprehension.gpt2.data <- comprehension.gpt2.data %>%
  mutate(verb_type = fct_relevel(verb_type),
         rc_type = fct_relevel(rc_type))
contrasts(comprehension.gpt2.data$verb_type)=contr.sum(2)
contrasts(comprehension.gpt2.data$rc_type)=contr.sum(2)

gpt2_analysis <- lmer(logodds ~ verb_type * rc_type + 
                        (1+verb_type+rc_type|item_id),
                         comprehension.gpt2.data)
summary(gpt2_analysis)


# 4. Data (Complex question) ----
## Llama3.2-1B ----
comprehension.llama1B.complex.data <- read.csv("../../../data/comprehension_nonIC_full_llama/comprehension_nonIC_full_llama_Llama-3.2-1B_1.csv", header=TRUE) %>%
  na.omit()
comprehension.llama1B.complex.data <- comprehension.llama1B.complex.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

## Llama3.2-3B ----
comprehension.llama3B.complex.data <- read.csv("../../../data/comprehension_nonIC_full_llama/comprehension_nonIC_full_llama_Llama-3.2-3B_1.csv", header=TRUE) %>%
  na.omit()
comprehension.llama3B.complex.data <- comprehension.llama3B.complex.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

## GPT-2 ----
comprehension.gpt2.complex.data <- read.csv("../../../data/comprehension_nonIC_full_llama/comprehension_nonIC_full_llama_gpt2_1.csv", header=TRUE) %>%
  na.omit()
comprehension.gpt2.complex.data <- comprehension.gpt2.complex.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

## all models ----
comprehension.complex.data <- bind_rows(lst(comprehension.llama1B.complex.data,comprehension.llama3B.complex.data, comprehension.gpt2.complex.data), .id="model") %>% 
  mutate(model=case_when(model == "comprehension.llama1B.complex.data" ~ "Llama3.2-1B",
                         model == "comprehension.llama3B.complex.data" ~ "Llama3.2-3B",
                         model == "comprehension.gpt2.complex.data" ~ "GPT2"))

## all models (mean) ----
comprehension_complex_mean <- comprehension.complex.data %>% 
  select(-c(Yes, No)) %>% 
  group_by(model,verb_type, continuation_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

# 5. Plot (Complex question) ----
## combined ----
### density of log odds ----
comprehension_complex_graph <- ggplot(comprehension.complex.data,
                              aes(x=logodds,
                                  fill=verb_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  # annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
  #                                                length=unit(2,"mm")), 
  #                                    gp=gpar(col="black", lwd=1.5)), xmin = -5.3, xmax = -0.4, ymin = -0.16, ymax = -0.16) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer exp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -0.5, xmax = 0, ymin = -0.17, ymax = -0.17) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer nonexp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -6.8, xmax = -6.3, ymin = -0.17, ymax = -0.17) +
  coord_cartesian(clip="off") +
  facet_grid(model ~ .) +
  labs(y = "density",
       x = "log odds (yes vs. no)")
comprehension_complex_graph
ggsave(comprehension_complex_graph, file="../graphs/comprehension_nonIC_sent_complex_llama.pdf", width=8, height=4)


### bar graph for means ----
comprehension_complex_bar_graph <- ggplot(comprehension_complex_mean,
                                  aes(x=verb_type,y=Mean,
                                      fill=verb_type,
                                      pattern=continuation_type)) +
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
  labs(y = "Mean log probability of the answer",
       x = "Answer Type") +
  facet_wrap(. ~ model) +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10)) +
  scale_fill_brewer(palette = "Dark2",guide="none") 
comprehension_complex_bar_graph
ggsave(comprehension_complex_bar_graph, file="../graphs/comprehension_nonIC_sent_complex_llama_mean.pdf", width=7, height=4)

# 6. Data (No before Yes) ----
## Llama3.2-1B ----
comprehension.no.llama1B.data <- read.csv("../../../data/comprehension_nonIC_full_alt_llama_no/comprehension_nonIC_full_alt_llama_no_Llama-3.2-1B_1.csv", header=TRUE) %>%
  na.omit()
comprehension.no.llama1B.data <- comprehension.no.llama1B.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

## Llama3.2-3B ----
comprehension.no.llama3B.data <- read.csv("../../../data/comprehension_nonIC_full_alt_llama/comprehension_nonIC_full_alt_llama_Llama-3.2-3B_1.csv", header=TRUE) %>%
  na.omit()
comprehension.no.llama3B.data <- comprehension.no.llama3B.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

## GPT-2 ----
comprehension.no.gpt2.data <- read.csv("../../../data/comprehension_nonIC_full_alt_llama/comprehension_nonIC_full_alt_llama_gpt2_1.csv", header=TRUE) %>%
  na.omit()
comprehension.no.gpt2.data <- comprehension.no.gpt2.data %>% 
  select(item_id, verb, verb_type, continuation_type, answer, critical_prob) %>% 
  pivot_wider(names_from = answer, values_from = critical_prob) %>%
  mutate(logodds = Yes-No)

## all models ----
comprehension.no.data <- bind_rows(lst(comprehension.no.llama1B.data,comprehension.no.gpt2.data), .id="model") %>% 
  mutate(model=case_when(model == "comprehension.no.llama1B.data" ~ "Llama3.2-1B",
                         model == "comprehension.no.llama3B.data" ~ "Llama3.2-3B",
                         model == "comprehension.no.gpt2.data" ~ "GPT2"))

## all models (mean) ----
comprehension_no_mean <- comprehension.no.data %>% 
  select(-c(Yes, No)) %>% 
  group_by(model,verb_type, continuation_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

# 2. Plot ----
## combined ----
### density of log odds ----
comprehension_no_graph <- ggplot(comprehension.no.data,
                              aes(x=logodds,
                                  fill=verb_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  # annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
  #                                                length=unit(2,"mm")),
  #                                    gp=gpar(col="black", lwd=1.5)), xmin = -1.1, xmax = 1.7, ymin = -0.62, ymax = -0.62) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer Yes", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = 1.5, xmax = 2.0, ymin = -0.68, ymax = -0.68) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer No", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -1.9, xmax = -0.9, ymin = -0.68, ymax = -0.68) +
  # coord_cartesian(clip="off") +
  facet_grid(model ~ .) +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size=12),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size=12)) +
  labs(y = "density",
       x = "log odds (yes vs. no)")
comprehension_no_graph
