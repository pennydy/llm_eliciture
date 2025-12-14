library(lme4)
library(lmerTest)
library(brms)
library(dplyr)
library(emmeans)
library(tidyverse)
library(ggplot2)
library(ggsignif)
library(tidytext)
library(ggpattern)
library(RColorBrewer)
library(stringr)
library(grid)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")


# 1. Data ----
## Llama3.2-1B-Instruct ----
comprehension.llama1B.data <- read.csv("../../../data/comprehension_nonIC_full_why/comprehension_nonIC_full_why_Llama-3.2-1B-Instruct_generate_2_annotate.csv", header=TRUE) %>%
  na.omit()

comprehension.llama1B.data <- comprehension.llama1B.data %>% 
  select(item_id, verb, verb_type, rc_type, annotated_answer, intended, error_type)

comprehension_llama1B_response <- subset(comprehension.llama1B.data, annotated_answer != "else")
comprehension_llama1B_exclude <- subset(comprehension.llama1B.data, annotated_answer == "else")

# comprehension.llama1B.response.type <- comprehension.llama1B.data %>% 
#   mutate(error_type = if_else(intended=="yes", "intended", error_type)) %>% 
#   mutate(error_type = if_else(annotated_answer == "else", "others", error_type))

# comprehension.llama1B.response.type.mean <- comprehension.llama1B.response.type %>% 
#   group_by(error_type) %>% 
#   summarize(proportion = n()/nrow(comprehension.llama1B.response.type),
#             CILow = ci.low(proportion),
#             CIHigh = ci.high(proportion))


comprehension.llama1B.response.type <- comprehension.llama1B.data %>% 
  mutate(error_type = if_else(intended=="yes", "intended_infer", error_type)) %>% 
  mutate(error_type = if_else(annotated_answer == "else", "others", error_type)) %>% 
  pivot_wider(names_from = error_type, values_from = annotated_answer) %>%
  rename(extended_rc = "extended inference, related to rc",
         extended_nonrc = "extended inference, not related to rc",
         rc = "rc as reason",
         idk = "idk in ic-exp condition") %>% 
  mutate(intended_infer = if_else(is.na(intended_infer), 0, 1),
         extended_rc = if_else(is.na(extended_rc), 0, 1),
         extended_nonrc = if_else(is.na(extended_nonrc), 0, 1),
         rc = if_else(is.na(rc), 0, 1),
         idk = if_else(is.na(idk), 0, 1),
         others = if_else(is.na(others), 0, 1))

comprehension.llama1B.response.type <- comprehension.llama1B.response.type %>% 
  pivot_longer(cols=c(intended_infer, extended_rc, extended_nonrc, rc, idk, others),
               values_to = "error_exist",
               names_to = "error_type")

# comprehension.llama1B.response.type <- comprehension.llama1B.data %>%
#   mutate(error_type = if_else(annotated_answer=="else", "others", error_type)) %>%
#   mutate(error_type = if_else(intended=="yes", "intended", error_type)) %>%
#   mutate(error_type = as.factor(error_type)) %>%
#   mutate(error_type = fct_recode(error_type, "extended_rc"="extended inference, related to rc", "extended_nonrc" = "extended inference, not related to rc", "rc" = "rc as reason", "idk" = "idk in ic-exp condition"))
# 
# comprehension.llama1B.response.type_mean <- comprehension.llama1B.response.type %>%
#   group_by(error_type) %>%
#   summarize(count = n(),
#             proportion = count/240,
#             CILow = ci.low(proportion),
#             CIHigh = ci.high(proportion)) %>%
#   ungroup() %>%
#   mutate(YMin=proportion-CILow,
#          YMax=proportion+CIHigh)

comprehension.llama1B.response.type_mean <- comprehension.llama1B.response.type %>%
  group_by(error_type) %>%
  summarize(Mean = mean(error_exist),
            CILow = ci.low(error_exist),
            CIHigh = ci.high(error_exist)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

comprehension_llama1B_mean <- comprehension_llama1B_response %>% 
  mutate(answer_numeric = if_else(annotated_answer=="yes", 1, 0)) %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(answer_numeric),
            CILow = ci.low(answer_numeric),
            CIHigh = ci.high(answer_numeric)) %>% 
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

comprehension_llama1B_accuracy_mean <- comprehension_llama1B_response %>% 
  mutate(intended_numeric = if_else(intended=="yes", 1, 0)) %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(intended_numeric),
            CILow = ci.low(intended_numeric),
            CIHigh = ci.high(intended_numeric)) %>% 
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)


## Llama3.2-3B-Instruct ----
comprehension.llama3B.data <- read.csv("../../../data/comprehension_nonIC_full_why_cont1/comprehension_nonIC_full_why_cont1_Llama-3.2-3B-Instruct_generate_2_annotated.csv", header=TRUE) %>%
  na.omit()

comprehension.llama3B.data <- comprehension.llama3B.data %>% 
  select(item_id, verb, verb_type, rc_type, annotated_answer, intended, error_type)

comprehension_llama3B_response <- subset(comprehension.llama3B.data, annotated_answer != "else")
comprehension_llama3B_exclude <- subset(comprehension.llama3B.data, annotated_answer == "else")

comprehension.llama3B.response.type <- comprehension.llama3B.data %>% 
  mutate(error_type = if_else(intended=="yes", "intended_infer", error_type),
         error_type = if_else(annotated_answer == "else", "others", error_type))

comprehension_llama3B_mean <- comprehension_llama3B_response %>% 
  mutate(answer_numeric = if_else(annotated_answer=="yes", 1, 0)) %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(answer_numeric),
            CILow = ci.low(answer_numeric),
            CIHigh = ci.high(answer_numeric)) %>% 
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)

comprehension_llama3B_accuracy_mean <- comprehension_llama3B_response %>% 
  mutate(intended_numeric = if_else(intended=="yes", 1, 0)) %>% 
  group_by(verb_type, rc_type) %>% 
  summarize(Mean = mean(intended_numeric),
            CILow = ci.low(intended_numeric),
            CIHigh = ci.high(intended_numeric)) %>% 
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh)


## all models ----
comprehension.data <- bind_rows(lst(comprehension_llama1B_response,comprehension_llama3B_response), .id="model") %>% 
  mutate(model=case_when(model == "comprehension_llama1B_response" ~ "Llama3.2-1B-Instruct",
                         model == "comprehension_llama3B_response" ~ "Llama3.2-3B-Instruct"))
# comprehension.llama1B.response.type_mean
comprehension_data_mean <- bind_rows(lst(comprehension_llama1B_mean,comprehension_llama3B_mean), .id="model") %>% 
  mutate(model=case_when(model == "comprehension_llama1B_mean" ~ "Llama3.2-1B-Instruct",
                         model == "comprehension_llama3B_mean" ~ "Llama3.2-3B-Instruct"))

comprehension.all.response.type <- comprehension.data %>% 
  mutate(error_type = if_else(intended=="yes", "intended_infer", error_type)) %>% 
  mutate(error_type = if_else(annotated_answer == "else", "others", error_type)) %>% 
  pivot_wider(names_from = error_type, values_from = annotated_answer) %>% 
  rename(extended_rc = "extended inference, related to rc",
         extended_nonrc = "extended inference, not related to rc",
         rc = "rc as reason",
         idk = "idk in ic-exp condition") %>% 
  mutate(intended_infer = if_else(is.na(intended_infer), 0, 1),
         extended_rc = if_else(is.na(extended_rc), 0, 1),
         extended_nonrc = if_else(is.na(extended_nonrc), 0, 1),
         rc = if_else(is.na(rc), 0, 1),
         idk = if_else(is.na(idk), 0, 1),
         others = if_else(is.na(others), 0, 1))

comprehension.response.type <- comprehension.all.response.type %>% 
  pivot_longer(cols=c(intended_infer, extended_rc, extended_nonrc, rc, idk, others),
               values_to = "error_exist",
               names_to = "error_type")

comprehension.response.type_mean <- comprehension.response.type %>%
  group_by(error_type, rc_type, verb_type, model) %>%
  summarize(Mean = mean(error_exist),
            CILow = ci.low(error_exist),
            CIHigh = ci.high(error_exist)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh) %>% 
  mutate(error_type = if_else(error_type == "intended_infer", "intended", error_type)) %>% 
  mutate(error_type = factor(error_type, levels=c("intended", "rc", "extended_rc", "extended_nonrc","idk", "others")))

# 2. Plot ----
## Llama3.2-1B-Instruct ----
### bar graph for means ----
comprehension_1B_bar_graph <- ggplot(comprehension_llama1B_mean,
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
  scale_fill_brewer(palette = "Dark2", guide="none") +
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  labs(y = "Proportion of yes",
       x = "Verb type") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10))
comprehension_1B_bar_graph

## Llama3.2-3B-Instruct ----
### bar graph for means ----
comprehension_3B_bar_graph <- ggplot(comprehension_llama3B_mean,
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
  scale_fill_brewer(palette = "Dark2", guide="none") +
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  labs(y = "Proportion of yes",
       x = "Verb type") +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10))
comprehension_3B_bar_graph

## all models ----
### bar graph for means ----
comprehension_bar_graph <- ggplot(comprehension_data_mean,
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
  scale_fill_brewer(palette = "Dark2", guide="none") +
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  facet_wrap(. ~ model) +
  labs(y = "Proportion of explanation answer",
       x = "Verb type") +
  theme(legend.position = "top",
        strip.text.x = element_text(size = 14),
        legend.text = element_text(size=12),
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 14),)
comprehension_bar_graph
ggsave(comprehension_bar_graph, file="../graphs/comprehension_nonIC_full_why_llama_instruct.pdf", width=8, height=4)

## answer type analysis ----
answer_type_graph <- ggplot(comprehension.response.type_mean,
                                  aes(x=error_type, y=Mean,
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
                position=position_dodge(width=0.9),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  scale_fill_brewer(palette = "Dark2") +
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  labs(y = "Response probability",
       x = "Response type") +
  facet_wrap(model~.) +
  theme(legend.position = "top",
        strip.text.x = element_text(size = 14),
        legend.text = element_text(size=12),
        axis.text.x = element_text(angle = 60, vjust = 0.5, hjust=0.5, size = 12),
        axis.title.x = element_text(size = 14),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 14))
answer_type_graph
ggsave(answer_type_graph, file="../graphs/comprehension_nonIC_full_why_llama_instruct_answer_type_by_condition.pdf", width=8, height=5)


# 3. Statistical analysis ----
## Llama3.2-1B-Instruct ----
comprehension.llama1B <- comprehension_llama1B_response %>% 
  mutate(rc_type = fct_relevel(rc_type,ref="exp"),
         verb_type = fct_relevel(verb_type,ref="IC"),
         annotated_answer = fct_relevel(annotated_answer,ref="yes"))

comprehension.llama1B.test <- comprehension.llama1B %>% 
  mutate(rc_type = if_else(rc_type == "exp", 0, 1),
         verb_type = if_else(verb_type == "IC", 0 ,1))
# contrasts(comprehension.llama1B$rc_type)=contr.sum(2)
# contrasts(comprehension.llama1B$verb_type)=contr.sum(2)
cor(comprehension.llama1B.test[c("rc_type", "verb_type")])

comprehension.llama1B.test$rc_type <- scale(comprehension.llama1B.test$rc_type, scale=FALSE)

comprehension.llama1B.test$verb_type <- scale(comprehension.llama1B.test$verb_type, scale=FALSE)

llama1B_analysis <- glmer(annotated_answer ~ rc_type * verb_type + (1+verb_type|item_id),
                          family = "binomial",
                          data=comprehension.llama1B,
                          glmerControl(optimizer = "bobyqa", optCtrl=list(maxfun=2e5)))
summary(llama1B_analysis)

llama1B_brms <- brm(annotated_answer ~ rc_type * verb_type + (1+rc_type+verb_type|item_id),
                     family = "bernoulli",
                     data=comprehension.llama1B,
                     control=list(max_treedepth = 15, adapt_delta = 0.99))
post_samples = data.frame(fixef(llama1B_brms, summary = F))
sum(post_samples$verb_type1 > 0) / length(post_samples$verb_type1)
fixef(llama1B_brms)

## Llama3.2-3B-Instruct ----
comprehension.llama3B <- comprehension_llama3B_response %>% 
  mutate(rc_type = fct_relevel(rc_type,ref="exp"),
         verb_type = fct_relevel(verb_type,ref="IC"),
         annotated_answer = fct_relevel(annotated_answer,ref="yes"))
# contrasts(comprehension.llama3B$rc_type)=contr.sum(2)
# contrasts(comprehension.llama3B$verb_type)=contr.sum(2)
llama3B_analysis <- glmer(annotated_answer ~ rc_type * verb_type + (1+verb_type+rc_type|item_id),
                          family = "binomial",
                          data = comprehension.llama3B,
                          glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
summary(llama3B_analysis)

llama3B_brms <- brm(annotated_answer ~ rc_type * verb_type + (1+rc_type+verb_type|item_id),
                    family = "bernoulli",
                    data=comprehension.llama3B,
                    control=list(max_treedepth = 15, adapt_delta = 0.99),
                    file="../cache/llama3B_instruct_comprehension")
post_samples_3B = data.frame(fixef(llama3B_brms, summary = F))
sum(post_samples_3B$verb_typenonIC > 0) / length(post_samples_3B$verb_typenonIC)
sum(post_samples_3B$rc_typenonexp > 0) / length(post_samples_3B$rc_typenonexp)
fixef(llama3B_brms)
