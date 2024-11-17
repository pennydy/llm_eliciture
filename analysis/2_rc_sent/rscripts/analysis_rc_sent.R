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

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
## Llama3.2-1B ----
rc_sent.llama1B.data <- read.csv("../../../data/rc_sent/rc_sent_Llama-3.2-1B_1.csv", header=TRUE) %>% 
  na.omit() 

rc_sent.llama1B.data <- rc_sent.llama1B.data %>% 
  select(item_id, verb, sentence_type, attachment, critical_prob) %>% 
  pivot_wider(names_from = attachment, values_from = critical_prob) %>% 
  mutate(logodds = high-low,
         high_prob = exp(high)/1+exp(high),
         low_prob = exp(low)/1+exp(low))

## Llama3.2-3B ----
rc_sent.llama3B.data <- read.csv("../../../data/rc_sent/rc_sent_Llama-3.2-3B_1.csv", header=TRUE) %>% 
  na.omit() 

rc_sent.llama3B.data <- rc_sent.llama3B.data %>% 
  select(item_id, verb, sentence_type, attachment, critical_prob) %>% 
  pivot_wider(names_from = attachment, values_from = critical_prob) %>% 
  mutate(logodds = high-low,
         high_prob = exp(high)/1+exp(high),
         low_prob = exp(low)/1+exp(low))

## gpt2 ----
rc_sent.gpt2.data <- read.csv("../../../data/rc_sent/rc_sent_gpt2_1.csv", header=TRUE) %>% 
  na.omit() 

rc_sent.gpt2.data <- rc_sent.gpt2.data %>% 
  select(item_id, verb, sentence_type, attachment, critical_prob) %>% 
  pivot_wider(names_from = attachment, values_from = critical_prob) %>% 
  mutate(logodds = high-low,
         high_prob = exp(high)/1+exp(high),
         low_prob = exp(low)/1+exp(low))


## all models ----
rc_sent.data <- bind_rows(lst(rc_sent.llama1B.data, rc_sent.llama3B.data,rc_sent.gpt2.data), .id="model") %>% 
  mutate(model=case_when(model == "rc_sent.llama1B.data" ~ "Llama3.2-1B",
                         model == "rc_sent.llama3B.data" ~ "Llama3.2-3B",
                         model == "rc_sent.gpt2.data" ~ "GPT2"))

# 2. Plot ----
## Llama3.2-1B ----
rc_sent_1B_graph <- ggplot(rc_sent.llama1B.data,
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
                                     gp=gpar(col="black", lwd=1.5)), xmin = -4.5, xmax = 1.5, ymin = -0.03, ymax = -0.03) +
  annotation_custom(grob = grid::textGrob(label = "Prefer high\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = 1.3, xmax = 1.8, ymin = -0.043, ymax = -0.043) +
  annotation_custom(grob = grid::textGrob(label = "Prefer low\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -5.6, xmax = -5.1, ymin = -0.043, ymax = -0.043) +
  # annotate("segment", 
  #            x = -5, xend = 2, y = -0.2, yend = -0.2,
  #            arrow=arrow(length=unit(0.3, "cm")))
  
  labs(y = "density",
       x = "log odds (high vs. low)")
rc_sent_1B_graph
ggsave(rc_sent_graph, file="../graphs/rc_sent_llama1B_1.pdf", width=8, height=4)

## Llama3.2-3B ----
rc_sent_3B_graph <- ggplot(rc_sent.llama3B.data,
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
                                     gp=gpar(col="black", lwd=1.5)), xmin = -4.3, xmax = 0.8, ymin = -0.035, ymax = -0.035) +
  annotation_custom(grob = grid::textGrob(label = "Prefer high\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = 0.8, xmax = 1.3, ymin = -0.043, ymax = -0.043) +
  annotation_custom(grob = grid::textGrob(label = "Prefer low\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -5.6, xmax = -5.1, ymin = -0.043, ymax = -0.043) +
  labs(y = "density",
       x = "log odds (high vs. low)")
rc_sent_3B_graph
ggsave(rc_sent_3B_graph, file="../graphs/rc_sent_llama3B_1.pdf", width=8, height=4)

## gpt2 ----
rc_sent_gpt2_graph <- ggplot(rc_sent.gpt2.data,
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
                                     gp=gpar(col="black", lwd=1.5)), xmin = -4.3, xmax = 1.8, ymin = -0.034, ymax = -0.034) +
  annotation_custom(grob = grid::textGrob(label = "Prefer high\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = 1.8, xmax = 2.3, ymin = -0.045, ymax = -0.045) +
  annotation_custom(grob = grid::textGrob(label = "Prefer low\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -5.6, xmax = -5.1, ymin = -0.045, ymax = -0.045) +
  labs(y = "density",
       x = "log odds (high vs. low)")
rc_sent_gpt2_graph
ggsave(rc_sent_gpt2_graph, file="../graphs/rc_sent_gpt2_1.pdf", width=8, height=4)

## combined ----
rc_sent_graph <- ggplot(rc_sent.data,
                           aes(x=logodds,
                               fill=sentence_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  # annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
  #                                                length=unit(2,"mm")), 
  #                                    gp=gpar(col="black", lwd=1.5)), xmin = -4.3, xmax = 0.8, ymin = -0.055, ymax = -0.055) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer high\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = 0.7, xmax = 1.2, ymin = -0.065, ymax = -0.065) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer low\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -5.6, xmax = -5.1, ymin = -0.065, ymax = -0.065) +
  annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
                                                 length=unit(2,"mm")), 
                                     gp=gpar(col="black", lwd=1.5)), xmin = -4.3, xmax = 1.8, ymin = -0.077, ymax = -0.077) +
  annotation_custom(grob = grid::textGrob(label = "Prefer high\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = 2, xmax = 2.5, ymin = -0.09, ymax = -0.09) +
  annotation_custom(grob = grid::textGrob(label = "Prefer low\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -5.6, xmax = -5.1, ymin = -0.09, ymax = -0.09) +
  coord_cartesian(clip="off") +
  facet_grid(model ~ .) +
  labs(y = "density",
       x = "log odds (high vs. low)")
rc_sent_graph
ggsave(rc_sent_graph, file="../graphs/rc_sent_2.pdf", width=8, height=4)


# 3. Statistical analysis ----
## Llama3.2-1B ----
llama1B_analysis <- lmer(logodds ~ sentence_type + (1|item_id),
                         rc_sent.llama1B.data)
summary(llama1B_analysis)

## Llama3.2-3B ----
llama3B_analysis <- lmer(logodds ~ sentence_type + (1|item_id),
                         rc_sent.llama3B.data)
summary(llama3B_analysis)


## gpt4 ----
gpt2_analysis <- lmer(logodds ~ sentence_type + (1|item_id),
                         rc_sent.gpt2.data)
summary(gpt2_analysis)
