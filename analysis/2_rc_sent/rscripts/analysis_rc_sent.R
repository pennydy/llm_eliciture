library(lme4)
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

## all models ----
rc_sent.data <- bind_rows(lst(rc_sent.llama1B.data, rc_sent.llama3B.data), .id="model") %>% 
  mutate(model=case_when(model == "rc_sent.llama1B.data" ~ "Llama3.2-1B",
                         model == "rc_sent.llama3B.data" ~ "Llama3.2-3B"))

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
       x = "log odds")
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
       x = "log odds")
rc_sent_3B_graph
ggsave(rc_sent_3B_graph, file="../graphs/rc_sent_llama3B_1.pdf", width=8, height=4)

## combined ----
rc_sent_graph <- ggplot(rc_sent.data,
                           aes(x=logodds,
                               fill=sentence_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
                                                 length=unit(2,"mm")), 
                                     gp=gpar(col="black", lwd=1.5)), xmin = -4.3, xmax = 0.8, ymin = -0.055, ymax = -0.055) +
  annotation_custom(grob = grid::textGrob(label = "Prefer high\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = 0.7, xmax = 1.2, ymin = -0.065, ymax = -0.065) +
  annotation_custom(grob = grid::textGrob(label = "Prefer low\nattachment", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -5.6, xmax = -5.1, ymin = -0.065, ymax = -0.065) +
  coord_cartesian(clip="off") +
  facet_grid(model ~ .) +
  labs(y = "density",
       x = "log odds")
rc_sent_graph
ggsave(rc_sent_graph, file="../graphs/rc_sent_1.jpg", width=8, height=4)


# 3. Statistical analysis ----
## gpt-3.5-turbo ----
rc_alt.gpt35.data <- rc_alt.gpt35.data %>% 
  mutate(sentence_type = fct_relevel(as.factor(sentence_type)),
         attachment = as.factor(attachment))
gpt35 <- glmer(attachment ~ sentence_type + (1|sent_id),
               family = "binomial",
               data=rc_alt.gpt35.data)
summary(gpt35)

## gpt-4 ----
rc_alt.gpt4.data <- rc_alt.gpt4.data %>% 
  mutate(sentence_type = fct_relevel(as.factor(sentence_type)),
         attachment = as.factor(attachment))
gpt4 <- glmer(attachment ~ sentence_type + (1|sent_id),
               family = "binomial",
               data=rc_alt.gpt4.data)
summary(gpt4)

## gpt-4o ----
rc_alt.gpt4o.data <- rc_alt.gpt4o.data %>% 
  mutate(sentence_type = fct_relevel(as.factor(sentence_type)),
         attachment = as.factor(attachment))
gpt4o <- glmer(attachment ~ sentence_type + (1|sent_id),
              family = "binomial",
              data=rc_alt.gpt4o.data)
summary(gpt4o)
