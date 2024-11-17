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

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")


# 1. Data ----
## Llama3.2-1B ----
comprehension.llama1B.data <- read.csv("../../../data/comprehension_sent/comprehension_sent_Llama-3.2-1B_1.csv", header=TRUE) %>% 
  na.omit() 

comprehension.llama1B.data <- comprehension.llama1B.data %>% 
  # filter(sentence_type == "IC") %>%
  select(item_id, verb, sentence_type, continuation_type, critical_prob) %>% 
  pivot_wider(names_from = continuation_type, values_from = critical_prob) %>% 
  mutate(logodds = exp-nonexp,
         exp_prob = exp(exp)/1+exp(exp),
         nonexp_prob = exp(nonexp)/1+exp(nonexp))

## Llama3.2-3B ----
comprehension.llama3B.data <- read.csv("../../../data/comprehension_sent/comprehension_sent_Llama-3.2-3B_1.csv", header=TRUE) %>% 
  na.omit() 

comprehension.llama3B.data <- comprehension.llama3B.data %>% 
  # filter(sentence_type == "IC") %>%
  select(item_id, verb, sentence_type, continuation_type, critical_prob) %>% 
  pivot_wider(names_from = continuation_type, values_from = critical_prob) %>% 
  mutate(logodds = exp-nonexp,
         exp_prob = exp(exp)/1+exp(exp),
         nonexp_prob = exp(nonexp)/1+exp(nonexp))

## gpt2 ----
comprehension.gpt2.data <- read.csv("../../../data/comprehension_sent/comprehension_sent_gpt2_1.csv", header=TRUE) %>% 
  na.omit() 

comprehension.gpt2.data <- comprehension.gpt2.data %>% 
  # filter(sentence_type == "IC") %>%
  select(item_id, verb, sentence_type, continuation_type, critical_prob) %>% 
  pivot_wider(names_from = continuation_type, values_from = critical_prob) %>% 
  mutate(logodds = exp-nonexp,
         exp_prob = exp(exp)/1+exp(exp),
         nonexp_prob = exp(nonexp)/1+exp(nonexp))

## all models ----
comprehension.data <- bind_rows(lst(comprehension.llama1B.data,comprehension.llama3B.data, comprehension.gpt2.data), .id="model") %>% 
  mutate(model=case_when(model == "comprehension.llama1B.data" ~ "Llama3.2-1B",
                         model == "comprehension.llama3B.data" ~ "Llama3.2-3B",
                         model == "comprehension.gpt2.data" ~ "GPT2"))

# 2. Plot ----
## Llama3.2-1B ----
comprehension_1B_graph <- ggplot(comprehension.llama1B.data,
                           aes(x=logodds,
                               fill=sentence_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  coord_cartesian(clip="off") +
  annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
                                                 length=unit(2,"mm")), 
                                     gp=gpar(col="black", lwd=1.5)), xmin = -5.5, xmax = -0.5, ymin = -0.048, ymax = -0.048) +
  annotation_custom(grob = grid::textGrob(label = "Prefer exp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -0.6, xmax = -0.1, ymin = -0.05, ymax = -0.05) +
  annotation_custom(grob = grid::textGrob(label = "Prefer nonexp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -6.8, xmax = -6.3, ymin = -0.05, ymax = -0.05) +
  labs(y = "density",
       x = "log odds (exp vs. nonexp)")
comprehension_1B_graph
ggsave(comprehension_1B_graph, file="../graphs/comprehension_sent_llama1B-IC_1.pdf", width=8, height=4)

## Llama3.2-3B ----
comprehension_3B_graph <- ggplot(comprehension.llama3B.data,
                                 aes(x=logodds,
                                     fill=sentence_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  coord_cartesian(clip="off") +
  annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
                                                 length=unit(2,"mm")), 
                                     gp=gpar(col="black", lwd=1.5)), xmin = -4.8, xmax = -0.5, ymin = -0.04, ymax = -0.04) +
  annotation_custom(grob = grid::textGrob(label = "Prefer exp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -0.6, xmax = -0.1, ymin = -0.04, ymax = -0.04) +
  annotation_custom(grob = grid::textGrob(label = "Prefer nonexp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -6, xmax = -5.5, ymin = -0.04, ymax = -0.04) +
  labs(y = "density",
       x = "log odds (exp vs. nonexp)")
comprehension_3B_graph
ggsave(comprehension_3B_graph, file="../graphs/comprehension_sent_llama3B-IC_1.pdf", width=8, height=4)

## gpt2 ----
comprehension_gpt2_graph <- ggplot(comprehension.gpt2.data,
                                 aes(x=logodds,
                                     fill=sentence_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  coord_cartesian(clip="off") +
  annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
                                                 length=unit(2,"mm")), 
                                     gp=gpar(col="black", lwd=1.5)), xmin = -4.3, xmax = -0.5, ymin = -0.08, ymax = -0.08) +
  annotation_custom(grob = grid::textGrob(label = "Prefer exp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -0.6, xmax = -0.1, ymin = -0.085, ymax = -0.085) +
  annotation_custom(grob = grid::textGrob(label = "Prefer nonexp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -5.4, xmax = -4.9, ymin = -0.085, ymax = -0.085) +
  labs(y = "density",
       x = "log odds (exp vs. nonexp)")
comprehension_gpt2_graph
ggsave(comprehension_gpt2_graph, file="../graphs/comprehension_sent_gpt2-IC_1.pdf", width=8, height=4)

## combined ----
comprehension_graph <- ggplot(comprehension.data,
                        aes(x=logodds,
                            fill=sentence_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
                                                 length=unit(2,"mm")), 
                                     gp=gpar(col="black", lwd=1.5)), xmin = -5.3, xmax = -0.4, ymin = -0.16, ymax = -0.16) +
  annotation_custom(grob = grid::textGrob(label = "Prefer exp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -0.5, xmax = 0, ymin = -0.17, ymax = -0.17) +
  annotation_custom(grob = grid::textGrob(label = "Prefer nonexp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -6.8, xmax = -6.3, ymin = -0.17, ymax = -0.17) +
  coord_cartesian(clip="off") +
  facet_grid(model ~ .) +
  labs(y = "density",
       x = "log odds (exp vs. nonexp)")
comprehension_graph
ggsave(comprehension_graph, file="../graphs/comprehension_sent_1.pdf", width=8, height=4)


# 3. Statistical analysis ----
## Llama3.2-1B ----
llama1B_analysis <- lmer(logodds ~ sentence_type + (1|item_id),
                    comprehension.llama1B.data)
summary(llama1B_analysis)

## Llama3.2-3B ----
llama3B_analysis <- lmer(logodds ~ sentence_type + (1|item_id),
                         comprehension.llama3B.data)
summary(llama3B_analysis)

## gpt2 ----
gpt2_analysis <- lmer(logodds ~ sentence_type + (1|item_id),
                         comprehension.gpt2.data)
summary(gpt2_analysis)
