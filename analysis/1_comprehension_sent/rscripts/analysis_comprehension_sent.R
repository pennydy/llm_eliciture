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
neg_patterns <- c("No", "does not", "not")
# files for llama3.2-1B/3B/gpt are not correct
## Llama3.2-1B-Instruct ----
comprehension.llama1B.instruct.data <- read.csv("../../../data/comprehension_nonIC_full_alt/comprehension_nonIC_full_alt_Llama-3.2-1B-Instruct_generate_1.csv", header=TRUE) %>% 
  na.omit() 
comprehension.llama1B.instruct.data <- comprehension.llama1B.instruct.data %>% 
  rename(rc_type = "continuation_type") %>% 
  mutate(answer_response=ifelse(grepl(paste(neg_patterns, collapse="|"), answer, ignore.case = TRUE), "no", "yes"))

sum(comprehension.llama1B.instruct.data$answer_response == "no")

## Llama3.2-3B-Instruct ----
comprehension.llama3B.instruct.data <- read.csv("../../../data/comprehension_nonIC_full_alt/comprehension_nonIC_full_alt_Llama-3.2-3B-Instruct_generate_1.csv", header=TRUE) %>% 
  na.omit() 
comprehension.llama3B.instruct.data <- comprehension.llama3B.instruct.data %>% 
  rename(rc_type = "continuation_type") %>% 
  mutate(answer_response=ifelse(grepl(paste(neg_patterns, collapse="|"), answer, ignore.case = TRUE), "no", "yes"))

sum(comprehension.llama3B.instruct.data$answer_response == "no")

comprehension_data <- bind_rows(lst(comprehension.llama1B.instruct.data, comprehension.llama3B.instruct.data), .id="model") %>% 
  mutate(model = case_when(model == "comprehension.llama1B.instruct.data" ~ "Llama1B-Instruct",
                           model == "comprehension.llama3B.instruct.data" ~ "Llama3B-Instruct")) %>% 
  select(model, verb_type, rc_type, answer_response, item_id)

comprehension_means <- comprehension_data %>% 
  mutate(answer_numerical = ifelse(answer_response=="no", 0, 1)) %>% 
  group_by(model, verb_type, rc_type) %>% 
  summarize(Mean = mean(answer_numerical),
            CILow = ci.low(answer_numerical),
            CIHigh = ci.high(answer_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

# 2. Data ----
# comprehension_graph <- ggplot(comprehension_means,
#                                   aes(x=verb_type,y=Mean,
#                                       fill=verb_type,
#                                       pattern=rc_type)) +
#   geom_bar_pattern(
#     position = "dodge",
#     stat="identity",
#     pattern_angle = 45,
#     pattern_spacing = 0.02,
#     pattern_fill="black",
#     pattern_alpha=0.4,
#     alpha=0.7) +
#   geom_errorbar(aes(ymin=YMin,ymax=YMax),
#                 position=position_dodge(width=0.8),
#                 width=.2, 
#                 show.legend = FALSE) +
#   scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
#                        name="RC Type") +
#   theme_bw() +
#   labs(y = "Proportion of explanation answers",
#        x = "Verb Type") +
#   facet_wrap(. ~ model) +
#   theme(legend.position="top",
#         axis.text.x = element_text(size = 10),
#         axis.text.y = element_text(size = 10)) +
#   ylim(0,1) +
#   scale_fill_brewer(palette = "Dark2",guide="none")
# comprehension_graph
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
