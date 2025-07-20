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
library(brms)

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
  rename(verb_type = "sentence_type") %>% 
  pivot_wider(names_from = attachment, values_from = critical_prob) %>% 
  mutate(logodds = high-low,
         surp_high = -high,
         surp_low = -low,
         surp_diff = surp_high-surp_low)

## Llama3.2-1B-instruct ----
rc_sent.llama1B.instruct.data <- read.csv("../../../data/rc_sent/rc_sent_Llama-3.2-1B-Instruct_1.csv", header=TRUE) %>% 
  na.omit() 

rc_sent.llama1B.instruct.data <- rc_sent.llama1B.instruct.data %>% 
  select(item_id, verb, sentence_type, attachment, critical_prob) %>% 
  rename(verb_type = "sentence_type") %>% 
  pivot_wider(names_from = attachment, values_from = critical_prob) %>% 
  mutate(logodds = high-low,
         surp_high = -high,
         surp_low = -low,
         surp_diff = surp_high-surp_low)

## Llama3.2-3B ----
rc_sent.llama3B.data <- read.csv("../../../data/rc_sent/rc_sent_Llama-3.2-3B_1.csv", header=TRUE) %>% 
  na.omit() 

rc_sent.llama3B.data <- rc_sent.llama3B.data %>% 
  select(item_id, verb, sentence_type, attachment, critical_prob) %>% 
  rename(verb_type = "sentence_type") %>% 
  pivot_wider(names_from = attachment, values_from = critical_prob) %>% 
  mutate(logodds = high-low,
         surp_high = -high,
         surp_low = -low,
         surp_diff = surp_high-surp_low)


## Llama3.2-3B-Instruct ----
rc_sent.llama3B.instruct.data <- read.csv("../../../data/rc_sent/rc_sent_Llama-3.2-3B-Instruct_1.csv", header=TRUE) %>% 
  na.omit() 

rc_sent.llama3B.instruct.data <- rc_sent.llama3B.instruct.data %>% 
  select(item_id, verb, sentence_type, attachment, critical_prob) %>% 
  rename(verb_type = "sentence_type") %>% 
  pivot_wider(names_from = attachment, values_from = critical_prob) %>% 
  mutate(logodds = high-low,
         surp_high = -high,
         surp_low = -low,
         surp_diff = surp_high-surp_low)


## gpt2 ----
rc_sent.gpt2.data <- read.csv("../../../data/rc_sent/rc_sent_gpt2_2.csv", header=TRUE) %>% 
  na.omit() 

rc_sent.gpt2.data <- rc_sent.gpt2.data %>% 
  select(item_id, verb, sentence_type, attachment, critical_prob) %>% 
  rename(verb_type = "sentence_type") %>% 
  pivot_wider(names_from = attachment, values_from = critical_prob) %>% 
  mutate(logodds = high-low,
         surp_high = -high,
         surp_low = -low,
         surp_diff = surp_high-surp_low)


## all models ----
rc_sent.data <- bind_rows(lst(rc_sent.llama1B.data,rc_sent.llama1B.instruct.data, rc_sent.llama3B.data,rc_sent.llama3B.instruct.data,rc_sent.gpt2.data), .id="model") %>% 
  mutate(model=case_when(model == "rc_sent.llama1B.data" ~ "1B",
                         model == "rc_sent.llama1B.instruct.data" ~ "1B-Instruct",
                         model == "rc_sent.llama3B.data" ~ "3B",
                         model == "rc_sent.llama3B.instruct.data" ~ "3B-Instruct",
                         model == "rc_sent.gpt2.data" ~ "GPT2"))

# mean of log prob difference
rc_sent_mean <- rc_sent.data %>% 
  group_by(model, verb_type) %>% 
  summarize(Mean = mean(logodds),
            CILow = ci.low(logodds),
            CIHigh = ci.high(logodds)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

# mean of surp difference
rc_sent_surp_mean <- rc_sent.data %>% 
  group_by(model, verb_type) %>% 
  summarize(Mean = mean(surp_diff),
            CILow = ci.low(surp_diff),
            CIHigh = ci.high(surp_diff)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

# 2. Plot ----
## Llama3.2-1B ----
rc_sent_1B_graph <- ggplot(rc_sent.llama1B.data,
                       aes(x=logodds,
                           fill=verb_type)) +
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
                               fill=verb_type)) +
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
                               fill=verb_type)) +
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
## line graph for diff ----
rc_line_graph <- ggplot(rc_sent_mean,
                        aes(x=verb_type,y=Mean)) +
  geom_point(aes(color=verb_type),
             stat="identity",
             alpha=0.7,
             size=2) +
  geom_hline(yintercept=0, linetype="dashed", color = "grey") +
  geom_line(aes(group=1),linetype="dotted") +
  geom_errorbar(aes(ymin=YMin,ymax=YMax,color=verb_type),
                width=.2, 
                show.legend = FALSE) +
  scale_color_brewer(palette = "Dark2", guide="none") +
  theme_bw() +
  facet_grid(. ~ model) + 
  labs(y = "Mean log probability\ndifference (high - low)",
       x = "Verb type") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 12),
        axis.title.y = element_text(size = 12))
rc_line_graph
ggsave(rc_line_graph, file="../graphs/rc_sent_line_graph.pdf", width=8, height=4)


# rc_sent.data$model <- factor(rc_sent.data$model, levels=c("Llama3.2-1B", "Llama3.2-1B-Instruct", "Llama3.2-3B", "Llama3.2-3B-Instruct", "GPT2"))
# rc_sent.data$model <- factor(rc_sent.data$model, levels=c("GPT2","Llama3.2-1B", "Llama3.2-1B-Instruct", "Llama3.2-3B", "Llama3.2-3B-Instruct")) 
# rc_sent.data$model <- factor(rc_sent.data$model, levels=c("GPT2","1B", "1B-Instruct", "3B", "3B-Instruct")) 
rc_sent_graph <- ggplot(rc_sent.data,
                           aes(x=logodds,
                               fill=verb_type)) +
  geom_density(alpha=0.6) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  geom_vline(xintercept = 0, linetype="dotted") +
  labs(fill = "Verb type") +
  coord_cartesian(clip="off") +
  facet_wrap(~model, nrow=2) +
  facet_grid(model~.) +
  theme(strip.text.y = element_text(size = 9),
        axis.title.x = element_text(size=14),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size=14),
        axis.text.y = element_text(size = 12),
        legend.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.position = "top")+
        # legend.justification = "right",
        # legend.direction = "vertical") +
  labs(y = "density",
       x = "log odds (high vs. low)")
rc_sent_graph
ggsave(rc_sent_graph, file="../graphs/rc_sent_all_models_2.pdf", width=7, height=5)


# 3. Statistical analysis ----
## Llama3.2-1B ----
llama1B_analysis <- lmer(logodds ~ verb_type + (1|item_id),
                         rc_sent.llama1B.data)
summary(llama1B_analysis)

llama1B_bayesian_default_prior <- brm(logodds ~ verb_type + (1|item_id),
                                    data=rc_sent.llama1B.data,
                                    iter=8000,
                                    warmup = 4000,
                                    chains=4,
                                    cores=4,
                                    control=list(max_treedepth = 15, adapt_delta = 0.99),
                                    file="../cache/llama1B_default_prior",
                                    seed=1024)
summary(llama1B_bayesian_default_prior)

## Llama3.2-1B-Instruct ----
llama1B_instruct_analysis <- lmer(logodds ~ verb_type + (1|item_id),
                         rc_sent.llama1B.instruct.data)
summary(llama1B_instruct_analysis)

llama1B_instruct_bayesian_default_prior <- brm(logodds ~ verb_type + (1|item_id),
                                      data=rc_sent.llama1B.instruct.data,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/llama1B_instruct_default_prior",
                                      seed=1024)
summary(llama1B_instruct_bayesian_default_prior)

## Llama3.2-3B ----
llama3B_analysis <- lmer(logodds ~ verb_type + (1|item_id),
                         rc_sent.llama3B.data)
summary(llama3B_analysis)

llama3B_bayesian_default_prior <- brm(logodds ~ verb_type + (1|item_id),
                                      data=rc_sent.llama3B.data,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/llama3B_default_prior",
                                      seed=1024)
summary(llama3B_bayesian_default_prior)

## Llama3.2-3B-Instruct ----
rc_sent.llama3B.instruct.data <- rc_sent.llama3B.instruct.data %>%
  mutate(verb_type=fct_relevel(verb_type, "IC"))

# contrasts(rc_sent.llama3B.instruct.data$verb_type)=contr.sum(2)
llama3B_instruct_analysis <- lmer(logodds ~ verb_type + (1|item_id),
                         rc_sent.llama3B.instruct.data)
summary(llama3B_instruct_analysis)

llama3B_instruct_bayesian_default_prior <- brm(logodds ~ verb_type + (1|item_id),
                                      data=rc_sent.llama3B.instruct.data,
                                      iter=8000,
                                      warmup = 4000,
                                      chains=4,
                                      cores=4,
                                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                                      file="../cache/llama3B_instruct_default_prior",
                                      seed=1024)
summary(llama3B_instruct_bayesian_default_prior)

## gpt2 ----
gpt2_analysis <- lmer(logodds ~ verb_type + (1|item_id),
                         rc_sent.gpt2.data)
summary(gpt2_analysis)

gpt2_bayesian_default_prior <- brm(logodds ~ verb_type + (1|item_id),
                                               data=rc_sent.gpt2.data,
                                               iter=8000,
                                               warmup = 4000,
                                               chains=4,
                                               cores=4,
                                               control=list(max_treedepth = 15, adapt_delta = 0.99),
                                               file="../cache/gpt2_default_prior",
                                               seed=1024)
summary(gpt2_bayesian_default_prior)

# 4. Exploratory analyses ----
## Llama3.2-1B ----
rc_sent.llama1B.data.logprobs <- rc_sent.llama1B.data %>% 
  select(-c(logodds, surp_high, surp_low, surp_diff)) %>% 
  pivot_longer(cols=c(low, high),
               names_to="attachment",
               values_to="logprobs") %>% 
  mutate(verb_type=factor(verb_type),
         attachment=factor(attachment))

# rc_sent.llama1B.summary.logprobs <- rc_sent.llama1B.data.logprobs %>% 
#   group_by(verb_type, attachment) %>% 
#   summarize(Mean = mean(logprobs),
#             CILow = ci.low(logprobs),
#             CIHigh = ci.high(logprobs)) %>% 
#   ungroup() %>% 
#   mutate(YMin = Mean-CILow,
#          YMax = Mean+CIHigh)

contrasts(rc_sent.llama1B.data.logprobs$verb_type) = contr.sum(2)
contrasts(rc_sent.llama1B.data.logprobs$attachment) = contr.sum(2)
llama1B_analysis_exp <- lmer(logprobs ~ verb_type*attachment + (1|item_id),
                             rc_sent.llama1B.data.logprobs)
summary(llama1B_analysis_exp)
llama1B_emms <- emmeans(llama1B_analysis_exp,specs = ~ verb_type | attachment)
pairs(llama1B_emms)

## Llama3.2-1B-Instruct ----
rc_sent.llama1B.instruct.data.logprobs <- rc_sent.llama1B.instruct.data %>% 
  select(-c(logodds, surp_high, surp_low, surp_diff)) %>% 
  pivot_longer(cols=c(low, high),
               names_to="attachment",
               values_to="logprobs") %>% 
  mutate(verb_type=factor(verb_type),
         attachment=factor(attachment))

contrasts(rc_sent.llama1B.instruct.data.logprobs$verb_type) = contr.sum(2)
contrasts(rc_sent.llama1B.instruct.data.logprobs$attachment) = contr.sum(2)
llama1B_instruct_analysis_exp <- lmer(logprobs ~ verb_type*attachment + (1|item_id),
                             rc_sent.llama1B.instruct.data.logprobs)
summary(llama1B_instruct_analysis_exp)
llama1B_instruct_emms <- emmeans(llama1B_instruct_analysis_exp,specs = ~ attachment | verb_type)
pairs(llama1B_instruct_emms)

## Llama3.2-3B ----
rc_sent.llama3B.data.logprobs <- rc_sent.llama3B.data %>% 
  select(-c(logodds, surp_high, surp_low, surp_diff)) %>% 
  pivot_longer(cols=c(low, high),
               names_to="attachment",
               values_to="logprobs") %>% 
  mutate(verb_type=factor(verb_type),
         attachment=factor(attachment))

contrasts(rc_sent.llama3B.data.logprobs$verb_type) = contr.sum(2)
contrasts(rc_sent.llama3B.data.logprobs$attachment) = contr.sum(2)
llama3B_analysis_exp <- lmer(logprobs ~ verb_type + attachment + verb_type*attachment + (1|item_id),
                             rc_sent.llama3B.data.logprobs)
summary(llama3B_analysis_exp)
llama3B_emms <- emmeans(llama1B_analysis_exp,specs = ~ verb_type | attachment)
pairs(llama3B_emms)

## Llama3.2-3B-Instruct ----
rc_sent.llama3B.instruct.data.logprobs <- rc_sent.llama3B.instruct.data %>% 
  select(-c(logodds, surp_high, surp_low, surp_diff)) %>% 
  pivot_longer(cols=c(low, high),
               names_to="attachment",
               values_to="logprobs") %>% 
  mutate(verb_type=factor(verb_type),
         attachment=factor(attachment))

contrasts(rc_sent.llama3B.instruct.data.logprobs$verb_type) = contr.sum(2)
contrasts(rc_sent.llama3B.instruct.data.logprobs$attachment) = contr.sum(2)
llama3B_instruct_analysis_exp <- lmer(logprobs ~ verb_type*attachment + (1|item_id),
                                      rc_sent.llama3B.instruct.data.logprobs)
summary(llama3B_instruct_analysis_exp)
llama3B_instruct_emms <- emmeans(llama3B_instruct_analysis_exp,specs = ~ attachment | verb_type)
pairs(llama3B_instruct_emms)

## gpt2 ----
rc_sent.gpt2.data.logprobs <- rc_sent.gpt2.data %>% 
  select(-c(logodds, surp_high, surp_low, surp_diff)) %>% 
  pivot_longer(cols=c(low, high),
               names_to="attachment",
               values_to="logprobs")

gpt2_analysis_exp <- lmer(logprobs ~ verb_type*attachment + (1|item_id),
                          rc_sent.gpt2.data.logprobs)
summary(gpt2_analysis_exp)

## all models ----
rc_sent.data.logprobs <- bind_rows(lst(rc_sent.llama1B.data.logprobs,rc_sent.llama1B.instruct.data.logprobs, rc_sent.llama3B.data.logprobs,rc_sent.llama3B.instruct.data.logprobs,rc_sent.gpt2.data.logprobs), .id="model") %>% 
  mutate(model=case_when(model == "rc_sent.llama1B.data.logprobs" ~ "1B",
                         model == "rc_sent.llama1B.instruct.data.logprobs" ~ "1B-Instruct",
                         model == "rc_sent.llama3B.data.logprobs" ~ "3B",
                         model == "rc_sent.llama3B.instruct.data.logprobs" ~ "3B-Instruct",
                         model == "rc_sent.gpt2.data.logprobs" ~ "GPT2"))%>% 
  mutate(item_attachment=paste(item_id, attachment, sep="_"))

# mean of log prob difference
rc_sent_logprobs_mean <- rc_sent.data.logprobs %>% 
  group_by(model, verb_type, attachment) %>% 
  summarize(Mean = mean(logprobs),
            CILow = ci.low(logprobs),
            CIHigh = ci.high(logprobs)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh,
         condition=paste(verb_type, attachment, sep="_"))

# rc_sent_logprobs_graph <- ggplot(rc_sent_logprobs_mean %>% 
#                                    mutate(condition=fct_relevel(condition, "IC_high", "nonIC_high", "IC_low", "nonIC_low")),
rc_sent_logprobs_graph <- ggplot(rc_sent.data.logprobs %>% 
                                   mutate(condition=paste(verb_type, attachment, sep="_")),
                                 aes(x=condition,
                                     y=logprobs,
                                     fill=verb_type)) +
  geom_bar(data=rc_sent_logprobs_mean,
           aes(x=condition,
               y=Mean,
               fill=verb_type),
           alpha=0.6,
           position=position_dodge(),
           stat="identity") +
  geom_errorbar(data=rc_sent_logprobs_mean,
                aes(y=Mean,ymin=YMin,ymax=YMax),
                position=position_dodge(width=0.9),
                width=.2, 
                show.legend = FALSE) +
  geom_point(position=position_dodge(width=0.9),
             alpha=0.4)+
  # geom_label_repel(rc_sent.data.logprobs,
  #                  aes(label=verb),
  #                  color="black",fill="white",
  #                  box.padding=0.1,
  #                  segment.size=0.2, nudge_x=0.16, direction="y")+
  geom_line(aes(group=verb),
            color = "black",
            alpha = 0.4,
            # size=0.4,
            linetype = "dashed") +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  labs(fill = "Verb type") +
  facet_wrap(~model, nrow=2) +
  facet_grid(model~.) +
  theme(strip.text.y = element_text(size = 9),
        axis.title.x = element_text(size=14),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size=14),
        axis.text.y = element_text(size = 12),
        legend.title = element_text(size=14),
        legend.text = element_text(size=12),
        legend.position = "top")+
  labs(y = "log probabilities",
       x = "condition type")
rc_sent_logprobs_graph
ggsave(rc_sent_logprobs_graph, file="../graphs/rc_sent_logprobs_graph_by_verb.pdf", width=7, height=5)
