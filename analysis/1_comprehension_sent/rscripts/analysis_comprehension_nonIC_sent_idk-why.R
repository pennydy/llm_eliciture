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
library(brms)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")


# 1. Data ----
## Llama3.2-1B ----
# comprehension.llama1B.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-1B_2.csv", header=TRUE) %>%
#   na.omit(),
comprehension.llama1B.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-1B_1.csv", header=TRUE) %>%
  na.omit()

comprehension.llama1B.data <- comprehension.llama1B.data %>% 
  rename(rc_type = "continuation_type") %>% 
  select(item_id, verb, verb_type, rc_type, critical_region_sum) %>%
  pivot_wider(names_from = rc_type, values_from = critical_region_sum) %>%
  # select(item_id, verb, verb_type, continuation_type, critical_prob) %>% 
  # pivot_wider(names_from = continuation_type, values_from = critical_prob) %>% 
  mutate(surp_exp = -exp,
         surp_nonexp = -nonexp,
         surp_diff = surp_exp - surp_nonexp,
         logodds = exp-nonexp)

## Llama3.2-1B-Instruct ----
# comprehension.llama1B.instruct.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-1B-Instruct_2.csv", header=TRUE) %>%
#   na.omit()
comprehension.llama1B.instruct.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-1B-Instruct_1.csv", header=TRUE) %>%
  na.omit()

comprehension.llama1B.instruct.data <- comprehension.llama1B.instruct.data %>% 
  rename(rc_type = "continuation_type") %>% 
  select(item_id, verb, verb_type, rc_type, critical_region_sum) %>%
  pivot_wider(names_from = rc_type, values_from = critical_region_sum) %>%
  mutate(surp_exp = -exp,
         surp_nonexp = -nonexp,
         surp_diff = surp_exp - surp_nonexp,
         logodds = exp-nonexp)

## Llama3.2-3B ----
# comprehension.llama3B.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-3B_2.csv", header=TRUE) %>%
#   na.omit()
comprehension.llama3B.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-3B_1.csv", header=TRUE) %>%
  na.omit()

comprehension.llama3B.data <- comprehension.llama3B.data %>% 
  rename(rc_type = "continuation_type") %>%
  select(item_id, verb, verb_type, rc_type, critical_region_sum) %>%
  pivot_wider(names_from = rc_type, values_from = critical_region_sum) %>%
  # select(item_id, verb, verb_type, continuation_type, critical_prob) %>% 
  # pivot_wider(names_from = continuation_type, values_from = critical_prob) %>% 
  mutate(surp_exp = -exp,
         surp_nonexp = -nonexp,
         surp_diff = surp_exp - surp_nonexp,
         logodds = exp-nonexp)

## Llama3.2-3B-Instruct ----
# comprehension.llama3B.instruct.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-3B-Instruct_2.csv", header=TRUE) %>%
#   na.omit()
comprehension.llama3B.instruct.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_Llama-3.2-3B-Instruct_1.csv", header=TRUE) %>%
  na.omit()

comprehension.llama3B.instruct.data <- comprehension.llama3B.instruct.data %>% 
  rename(rc_type = "continuation_type") %>%
  select(item_id, verb, verb_type, rc_type, critical_region_sum) %>%
  pivot_wider(names_from = rc_type, values_from = critical_region_sum) %>%
  # select(item_id, verb, verb_type, continuation_type, critical_prob) %>% 
  # pivot_wider(names_from = continuation_type, values_from = critical_prob) %>% 
  mutate(surp_exp = -exp,
         surp_nonexp = -nonexp,
         surp_diff = surp_exp - surp_nonexp,
         logodds = exp-nonexp)

## gpt2 ----
# comprehension.gpt2.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_gpt2_2.csv", header=TRUE) %>%
#   na.omit()
comprehension.gpt2.data <- read.csv("../../../data/comprehension_nonIC_sent/comprehension_nonIC_sent_gpt2_1.csv", header=TRUE) %>%
  na.omit()

comprehension.gpt2.data <- comprehension.gpt2.data %>% 
  rename(rc_type = "continuation_type") %>%
  select(item_id, verb, verb_type, rc_type, critical_region_sum) %>%
  pivot_wider(names_from = rc_type, values_from = critical_region_sum) %>%
  # select(item_id, verb, verb_type, continuation_type, critical_prob) %>% 
  # pivot_wider(names_from = continuation_type, values_from = critical_prob) %>% 
  mutate(surp_exp = -exp,
         surp_nonexp = -nonexp,
         surp_diff = surp_exp - surp_nonexp,
         logodds = exp-nonexp)

## all models ----
comprehension.data <- bind_rows(lst(comprehension.llama1B.data,comprehension.llama1B.instruct.data,comprehension.llama3B.data,comprehension.llama3B.instruct.data,comprehension.gpt2.data), .id="model") %>% 
  mutate(model=case_when(model == "comprehension.llama1B.data" ~ "1B",
                         model == "comprehension.llama1B.instruct.data" ~ "1B-Instruct",
                         model == "comprehension.llama3B.data" ~ "3B",
                         model == "comprehension.llama3B.instruct.data" ~ "3B-Instruct",
                         model == "comprehension.gpt2.data" ~ "GPT2"))

## all models (mean) ----
comprehension_mean <- comprehension.data %>% 
  select(-c(logodds, surp_exp, surp_nonexp, surp_diff)) %>% 
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

## all models (mean surp) ----
comprehension_mean_surp <- comprehension.data %>% 
  select(-c(logodds, exp, nonexp, surp_diff)) %>% 
  pivot_longer(cols=c(surp_exp, surp_nonexp),
               names_to = "rc_type",
               values_to = "critical_region_surp") %>% 
  group_by(model,verb_type, rc_type) %>% 
  summarize(Mean = mean(critical_region_surp),
            CILow = ci.low(critical_region_surp),
            CIHigh = ci.high(critical_region_surp)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

## all models (diff) ----
comprehension_mean_diff <- comprehension.data %>% 
  group_by(model,verb_type) %>% 
  summarize(Mean = mean(surp_diff),
            CILow = ci.low(surp_diff),
            CIHigh = ci.high(surp_diff)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)
  
# 2. Plot ----
## Llama3.2-1B ----
comprehension_1B_graph <- ggplot(comprehension.llama1B.data,
                                 aes(x=logodds,
                                     fill=verb_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  coord_cartesian(clip="off") +
  # annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
  #                                                length=unit(2,"mm")), 
  #                                    gp=gpar(col="black", lwd=1.5)), xmin = -5.5, xmax = -0.5, ymin = -0.048, ymax = -0.048) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer exp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -0.6, xmax = -0.1, ymin = -0.05, ymax = -0.05) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer nonexp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -6.8, xmax = -6.3, ymin = -0.05, ymax = -0.05) +
  labs(y = "density",
       x = "log odds (exp vs. nonexp)")
comprehension_1B_graph
ggsave(comprehension_1B_graph, file="../graphs/comprehension_nonIC_sent_llama1B-IC_1.pdf", width=8, height=4)

## Llama3.2-3B ----
comprehension_3B_graph <- ggplot(comprehension.llama3B.data,
                                 aes(x=logodds,
                                     fill=verb_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  coord_cartesian(clip="off") +
  # annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
  #                                                length=unit(2,"mm")), 
  #                                    gp=gpar(col="black", lwd=1.5)), xmin = -4.8, xmax = -0.5, ymin = -0.04, ymax = -0.04) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer exp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -0.6, xmax = -0.1, ymin = -0.04, ymax = -0.04) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer nonexp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -6, xmax = -5.5, ymin = -0.04, ymax = -0.04) +
  labs(y = "density",
       x = "log odds (exp vs. nonexp)")
comprehension_3B_graph
ggsave(comprehension_3B_graph, file="../graphs/comprehension_nonIC_sent_llama3B-IC_1.pdf", width=8, height=4)

## gpt2 ----
comprehension_gpt2_graph <- ggplot(comprehension.gpt2.data,
                                   aes(x=logodds,
                                       fill=verb_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  coord_cartesian(clip="off") +
  # annotation_custom(grob = linesGrob(arrow=arrow(type="open", ends="both",
  #                                                length=unit(2,"mm")), 
  #                                    gp=gpar(col="black", lwd=1.5)), xmin = -4.3, xmax = -0.5, ymin = -0.08, ymax = -0.08) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer exp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -0.6, xmax = -0.1, ymin = -0.085, ymax = -0.085) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer nonexp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -5.4, xmax = -4.9, ymin = -0.085, ymax = -0.085) +
  labs(y = "density",
       x = "log odds (exp vs. nonexp)")
comprehension_gpt2_graph
ggsave(comprehension_gpt2_graph, file="../graphs/comprehension_nonIC_sent_gpt2-IC_1.pdf", width=8, height=4)

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
  #                                    gp=gpar(col="black", lwd=1.5)), xmin = -5.3, xmax = -0.4, ymin = -0.16, ymax = -0.16) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer exp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -0.5, xmax = 0, ymin = -0.17, ymax = -0.17) +
  # annotation_custom(grob = grid::textGrob(label = "Prefer nonexp", hjust=0, gp=gpar(col="black", cex=0.8)),xmin = -6.8, xmax = -6.3, ymin = -0.17, ymax = -0.17) +
  coord_cartesian(clip="off") +
  facet_grid(model ~ .) +
  theme(legend.position = "top",
        strip.text.x = element_text(size = 10),
        axis.title.x = element_text(size=14),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size=14),
        axis.text.y = element_text(size = 12),
        legend.title = element_text(size=14),
        legend.text = element_text(size=12)) +
  labs(y = "density",
       x = "log odds (exp - nonexp)")
comprehension_graph
ggsave(comprehension_graph, file="../graphs/comprehension_nonIC_sent_density_1.pdf", width=8, height=6)

### bar graph for means ----
comprehension_bar_graph <- ggplot(comprehension_mean_surp,
                                  aes(x=verb_type,
                                      y=Mean,
                                      fill=verb_type,
                                      pattern=rc_type)) +
  geom_bar_pattern( 
    # color="black",
    position = "dodge",
    stat="identity",
    pattern_angle = 45,
    pattern_spacing = 0.1,
    pattern_fill="black",
    pattern_alpha=0.6,
    alpha=0.7) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                position=position_dodge(width=0.8),
                width=.2, 
                show.legend = FALSE) +
  scale_pattern_manual(values=c(surp_exp = "stripe", surp_nonexp = "none"),
                       labels=c("exp","nonexp"),
                       name="RC Type") +
  theme_bw() +
  labs(y = "Mean surprisal of the continuation",
       x = "Verb Type") +
  facet_grid(. ~ model) +
  theme(legend.position = "top",
        strip.text.x = element_text(size = 9),
        axis.title.x = element_text(size=14),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size=14),
        axis.text.y = element_text(size = 12),
        legend.title = element_text(size=14),
        legend.text = element_text(size=12)) +
  scale_fill_brewer(palette = "Dark2", guide="none") 
comprehension_bar_graph
ggsave(comprehension_bar_graph, file="../graphs/comprehension_nonIC_sent_mean_surp_all_models_1.pdf", width=8, height=4)

### line graph for diff ----
comprehension_diff_graph <- ggplot(comprehension_mean_diff,
                                   aes(x=verb_type,
                                       y=Mean)) +
  geom_point(aes(color=verb_type))+
  geom_line(aes(group=1),linetype="dotted") +
  geom_errorbar(aes(ymin=YMin,ymax=YMax,color=verb_type),
                position="identity",
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  labs(y = "Mean surprisal difference of the\ncontinuation (exp - nonexp)",
       x = "Verb Type") +
  facet_grid(. ~ model) +
  scale_color_brewer(palette = "Dark2", guide="none") +
  theme(legend.position = "top",
        strip.text.x = element_text(size = 9),
        axis.title.x = element_text(size=14),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size=14),
        axis.text.y = element_text(size = 12),
        legend.title = element_text(size=14),
        legend.text = element_text(size=12)) +
  scale_fill_brewer(palette = "Dark2", guide="none") 
comprehension_diff_graph
ggsave(comprehension_diff_graph, file="../graphs/comprehension_nonIC_sent_diff_1.pdf", width=8, height=4)

# 3. Statistical analysis ----
# options(contrasts = c("contr.sum","contr.sum"))
## Llama3.2-1B ----
# using surprisal here, should be the same but opposite directions
comprehension_surp.llama1B <- comprehension.llama1B.data %>%
  select(c("item_id", "verb", "verb_type","surp_exp","surp_nonexp")) %>%
  pivot_longer(cols=c(surp_exp, surp_nonexp),
               names_to = "rc_type",
               values_to = "critical_region_surp") %>%
  mutate(rc_type=fct_relevel(rc_type,"surp_nonexp"),
         verb_type=fct_relevel(verb_type,"IC")) %>%
  na.omit() 

llama1B_surp_analysis <- lmer(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                              comprehension_surp.llama1B)
summary(llama1B_surp_analysis)
emmeans(llama1B_surp_analysis, pairwise ~ rc_type | verb_type,adjust = "bonferroni")

llama1B_bayesian_surp <- brm(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                      data=comprehension_surp.llama1B,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/llama1B_surp_default_prior_1",
                                      seed=1024)
summary(llama1B_bayesian_surp)
emmeans(llama1B_bayesian_surp, pairwise ~ rc_type | verb_type,adjust = "bonferroni")
prior_summary(llama1B_bayesian_surp)
fixef_data<-data.frame(fixef(llama1B_bayesian_surp, summary = F))
# colnames(fixef_data)
fixef_data %>%
  rename(interaction="rc_typesurp_exp.verb_typenonIC") %>%
  summarize(sum = sum("rc_type1">0),
            proportion = sum/n())

# using log probabilities
comprehension.llama1B <- comprehension.llama1B.data %>%
  select(c("item_id", "verb", "verb_type","exp","nonexp")) %>%
  pivot_longer(cols=c(exp, nonexp),
               names_to = "rc_type",
               values_to = "critical_region_logprob") %>%
  mutate(rc_type=fct_relevel(rc_type,"nonexp"),
         verb_type=fct_relevel(verb_type,"IC")) %>%
  na.omit() 

llama1B_analysis <- lmer(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                         comprehension.llama1B)
summary(llama1B_analysis)

llama1B_bayesian_default_prior <- brm(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                        data=comprehension.llama1B,
                        iter=8000,
                        warmup = 4000,
                        chains=4,
                        cores=4,
                        control=list(max_treedepth = 15, adapt_delta = 0.99),
                        file="../cache/llama1B_default_prior_1",
                        # file="../cache/llama1B_default_prior_dummy",
                        # file="../cache/llama1B_default_prior",
                        seed=1024)
summary(llama1B_bayesian_default_prior)
prior_summary(llama1B_bayesian_default_prior)
fixef_data<-data.frame(fixef(llama1B_bayesian_default_prior, summary = F))
# colnames(fixef_data)
fixef_data %>%
  rename(interaction="rc_type1.verb_type1") %>%
  summarize(sum = sum("rc_type1">0),
            proportion = sum/n())


## Llama3.2-1B-Instruct ----
# using surprisal
comprehension_surp.llama1B.instruct <- comprehension.llama1B.instruct.data %>%
  select(c("item_id", "verb", "verb_type","surp_exp","surp_nonexp")) %>%
  pivot_longer(cols=c(surp_exp, surp_nonexp),
               names_to = "rc_type",
               values_to = "critical_region_surp") %>%
  mutate(rc_type=fct_relevel(rc_type,"surp_nonexp"),
         verb_type=fct_relevel(verb_type,"IC")) %>%
  na.omit()

llama1B_instruct_surp_analysis <- lmer(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                       comprehension_surp.llama1B.instruct)
summary(llama1B_instruct_surp_analysis)
emmeans(llama1B_instruct_surp_analysis, pairwise ~ rc_type | verb_type,adjust = "bonferroni")

llama1B_instruct_bayesian_surp <- brm(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                               data=comprehension_surp.llama1B.instruct,
                                               iter=8000,
                                               warmup = 4000,
                                               chains=4,
                                               cores=4,
                                               control=list(max_treedepth = 15, adapt_delta = 0.99),
                                               file="../cache/llama1B_instruct_surp_default_prior_1",
                                               seed=1024)
summary(llama1B_instruct_bayesian_surp)
emmeans(llama1B_instruct_bayesian_surp, pairwise ~ rc_type | verb_type,adjust = "bonferroni")

# using log probabilities
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

llama1B_instruct_bayesian_default_prior <- brm(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                      data=comprehension.llama1B.instruct,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/llama1B_instruct_default_prior_1",
                                      # file="../cache/llama1B_instruct_default_prior_dummy",
                                      # file="../cache/llama1B_instruct_default_prior",
                                      seed=1024)
summary(llama1B_instruct_bayesian_default_prior)

hypothesis(llama1B_instruct_bayesian_default_prior,c('rc_type1 + rc_type1:verb_type1 = 0', 
                              'rc_type1 - rc_type1:verb_type1 = 0'))

## Llama3.2-3B ----
# using surprisal here, should be the same but opposite directions
comprehension_surp.llama3B <- comprehension.llama3B.data %>%
  select(c("item_id", "verb", "verb_type","surp_exp","surp_nonexp")) %>%
  pivot_longer(cols=c(surp_exp, surp_nonexp),
               names_to = "rc_type",
               values_to = "critical_region_surp") %>%
  mutate(rc_type=fct_relevel(rc_type,"surp_nonexp"),
         verb_type=fct_relevel(verb_type,"IC")) %>%
  na.omit() 

llama3B_surp_analysis <- lmer(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                              comprehension_surp.llama3B)
summary(llama3B_surp_analysis)
emmeans(llama3B_surp_analysis, pairwise ~ rc_type | verb_type,adjust = "bonferroni")

llama3B_bayesian_surp <- brm(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                             data=comprehension_surp.llama3B,
                             iter=8000,
                             warmup = 4000,
                             chains=4,
                             cores=4,
                             control=list(max_treedepth = 15, adapt_delta = 0.99),
                             file="../cache/llama3B_surp_default_prior_1",
                             seed=10)
summary(llama3B_bayesian_surp)
emmeans(llama3B_bayesian_surp, pairwise ~ rc_type | verb_type,adjust = "bonferroni")

prior_summary(llama3B_bayesian_surp)
fixef_data<-data.frame(fixef(llama1B_bayesian_surp, summary = F))
# colnames(fixef_data)
fixef_data %>%
  rename(interaction="rc_typesurp_exp.verb_typenonIC") %>%
  summarize(sum = sum("rc_type1">0),
            proportion = sum/n())

# using log probability
comprehension.llama3B <- comprehension.llama3B.data %>%
  select(c("item_id", "verb", "verb_type","exp","nonexp")) %>%
  pivot_longer(cols=c(exp, nonexp),
               names_to = "rc_type",
               values_to = "critical_region_logprob") %>%
  mutate(rc_type=fct_relevel(rc_type,ref="nonexp"),
         verb_type=fct_relevel(verb_type, ref="IC")) 

llama3B_analysis <- lmer(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                         comprehension.llama3B)
summary(llama3B_analysis)


llama3B_bayesian_default_prior <- brm(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                               data=comprehension.llama3B,
                                               iter=8000,
                                               warmup = 4000,
                                               chains=4,
                                               cores=4,
                                               control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/llama3B_default_prior_1",
                                      # file="../cache/llama3B_default_prior",
                                      # file="../cache/llama3B_default_prior_dummy",
                                               seed=10)
summary(llama3B_bayesian_default_prior)
fixef_data<-data.frame(fixef(llama3B_bayesian_default_prior, summary = F))
# colnames(fixef_data)
fixef_data %>%
  rename(interaction="rc_type1.verb_type1") %>%
  summarize(sum = sum(rc_type1>0),
            proportion = sum/n())


llama3B_bayesian <- brm(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                      data=comprehension.llama3B,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                        prior = c(set_prior("student_t(3, 0, 3)", class = "b"),
                                  set_prior("student_t(3, 0, 3)", class = "sd")),
                                      file="../cache/llama3B",
                                      seed=1024)
summary(llama3B_bayesian)
fixef_data<-data.frame(fixef(llama3B_bayesian, summary = F))
# colnames(fixef_data)
fixef_data %>%
  rename(interaction="rc_type1.verb_type1") %>%
  summarize(sum = sum(rc_type1>0),
            proportion = sum/n())

quantile(fixef_data$rc_type1,probs = c(0.025, 0.975)) 

fixef_data %>% 
  group_by(rc_type1) %>% 
  summarize(n())

hypothesis(llama3B_bayesian,c('rc_type1 + rc_type1:verb_type1 = 0', 
                              'rc_type1 - rc_type1:verb_type1 = 0'))

## Llama3.2-3B-Instruct ----
# using surprisal
comprehension_surp.llama3B.instruct <- comprehension.llama3B.instruct.data %>%
  select(c("item_id", "verb", "verb_type","surp_exp","surp_nonexp")) %>%
  pivot_longer(cols=c(surp_exp, surp_nonexp),
               names_to = "rc_type",
               values_to = "critical_region_surp") %>%
  mutate(rc_type=fct_relevel(rc_type,"surp_nonexp"),
         verb_type=fct_relevel(verb_type,"IC")) %>%
  na.omit()

llama3B_instruct_surp_analysis <- lmer(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                       comprehension_surp.llama3B.instruct)
summary(llama3B_instruct_surp_analysis)
emmeans(llama3B_instruct_surp_analysis, pairwise ~ rc_type | verb_type,adjust = "bonferroni")


llama3B_instruct_bayesian_surp <- brm(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                      data=comprehension_surp.llama3B.instruct,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/llama3B_instruct_surp_default_prior_1",
                                      seed=1024)
summary(llama3B_instruct_bayesian_surp)
emmeans(llama3B_instruct_bayesian_surp, pairwise ~ rc_type | verb_type,adjust = "bonferroni")

# using log probabilities
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

llama3B_instruct_bayesian_default_prior <- brm(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                      data=comprehension.llama3B.instruct,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/llama3B_instruct_default_prior_1",
                                      # file="../cache/llama3B_instruct_default_prior",
                                      # file="../cache/llama3B_instruct_default_prior_dummy",
                                      seed=1024)
summary(llama3B_instruct_bayesian_default_prior)

## gpt2 ----
# using surprisal
comprehension_surp.gpt2 <- comprehension.gpt2.data %>%
  select(c("item_id", "verb", "verb_type","surp_exp","surp_nonexp")) %>%
  pivot_longer(cols=c(surp_exp, surp_nonexp),
               names_to = "rc_type",
               values_to = "critical_region_surp") %>%
  mutate(rc_type=fct_relevel(rc_type,"surp_nonexp"),
         verb_type=fct_relevel(verb_type,"IC")) %>%
  na.omit()

gpt2_surp_analysis <- lmer(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                           comprehension_surp.gpt2)
summary(gpt2_surp_analysis)
emmeans(gpt2_surp_analysis, pairwise ~ rc_type | verb_type,adjust = "bonferroni")

gpt2_bayesian_surp <- brm(critical_region_surp ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                      data=comprehension_surp.gpt2,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/gpt2_surp_default_prior_1",
                                      seed=1024)
summary(gpt2_bayesian_surp)

comprehension.gpt2 <- comprehension.gpt2.data %>%
  select(c("item_id", "verb", "verb_type","exp","nonexp")) %>%
  pivot_longer(cols=c(exp, nonexp),
               names_to = "rc_type",
               values_to = "critical_region_logprob") %>%
  mutate(rc_type=fct_relevel(rc_type, "nonexp"),
         verb_type=fct_relevel(verb_type,"IC"))

# contrasts(comprehension.gpt2$continuation_type)=contr.sum(2)
# levels(comprehension.gpt2$continuation_type)
# gpt2_analysis <- lmer(critical_region_logprob ~ continuation_type + (1|item_id),
#                          comprehension.gpt2)
# summary(gpt2_analysis)
gpt2_analysis <- lmer(critical_region_logprob ~ rc_type * verb_type + (1+rc_type + verb_type|item_id),
                         comprehension.gpt2)
summary(gpt2_analysis)

gpt2_bayesian_default_prior <- brm(critical_region_logprob ~ rc_type * verb_type + (1+rc_type+ verb_type|item_id),
                                   data=comprehension.gpt2,
                                   iter=8000,
                                   warmup = 4000,
                                   chains=4,
                                   cores=4,
                                   control=list(max_treedepth = 15, adapt_delta = 0.99),
                                   file="../cache/gpt2_default_prior_1",
                                   # file="../cache/gpt2_default_prior",
                                   # file="../cache/gpt2_default_prior_dummy",
                                   seed=1024)
summary(gpt2_bayesian_default_prior)

fixef_data<-data.frame(fixef(gpt2_bayesian_default_prior, summary = F))
colnames(fixef_data)
fixef_data %>%
  rename(interaction="rc_typeexp.verb_typenonIC") %>%
  summarize(sum_rc = sum(rc_typeexp>0),
            proportion_rc = sum_rc/n(),
            sum_inter = sum(interaction>0),
            proportion_inter = sum_inter/n())

quantile(fixef_data$rc_typeexp,probs = c(0.025, 0.975)) 

fixef_data %>% 
  group_by(rc_typeexp) %>% 
  summarize(n())

hypothesis(gpt2_bayesian_default_prior,c('rc_type1 + rc_type1:verb_type1 = 0', 
                              'rc_type1 - rc_type1:verb_type1 = 0'))
