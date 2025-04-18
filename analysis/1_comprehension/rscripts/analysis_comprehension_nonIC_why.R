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
library(brms)
library(blme)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
## gpt-3.5-turbo ----
comprehension.gpt35.data <- read.csv("../../../data/comprehension_nonIC_full_why/comprehension_nonIC_full_why-gpt-3.5-turbo_1_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(annotated_answer = as.factor(tolower(annotated_answer)),
         intended = as.factor(tolower(intended)))

## gpt-4 ----
comprehension.gpt4.data <- read.csv("../../../data/comprehension_nonIC_full_why/comprehension_nonIC_full_why-gpt-4_1_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(annotated_answer = as.factor(tolower(annotated_answer)),
         intended = as.factor(tolower(intended)))

## gpt-4o ----
comprehension.gpt4o.data <- read.csv("../../../data/comprehension_nonIC_full_why/comprehension_nonIC_full_why-gpt-4o_1_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(annotated_answer = as.factor(tolower(annotated_answer)),
         intended = as.factor(tolower(intended)))

## Llama-3.2-1B-Instruct ----
comprehension.llama1B_instruct.data <- read.csv("../../../data/comprehension_nonIC_full_why/comprehension_nonIC_full_why_Llama-3.2-1B-Instruct_generate_2_annotate.csv", header=TRUE) 
sum(comprehension.llama1B_instruct.data$annotated_answer == "else")

comprehension.llama1B_instruct.data <- comprehension.llama1B_instruct.data %>% 
  # mutate(annotated_answer = ifelse(annotated_answer == "else", "no", annotated_answer)) %>% 
  na.omit() %>% 
  mutate(annotated_answer = as.factor(tolower(annotated_answer)),
         intended = as.factor(tolower(intended)))

## Llama-3.2-3B-Instruct ----
comprehension.llama3B_instruct.data <- read.csv("../../../data/comprehension_nonIC_full_why/comprehension_nonIC_full_why_cont1_Llama-3.2-3B-Instruct_generate_2_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(annotated_answer = as.factor(tolower(annotated_answer)),
         intended = as.factor(tolower(intended)))

## combined --- 
comprehension_all_data <- bind_rows(lst(comprehension.gpt4o.data, comprehension.gpt4.data, comprehension.gpt35.data, comprehension.llama1B_instruct.data, comprehension.llama3B_instruct.data), .id="model") %>% 
  mutate(model = case_when(model == "comprehension.gpt4o.data" ~ "gpt-4o",
                           model == "comprehension.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "comprehension.gpt4.data" ~ "gpt-4",
                           model == "comprehension.llama1B_instruct.data" ~ "llama1B_instruct",
                           model == "comprehension.llama3B_instruct.data" ~ "llama3B_instruct"))

comprehension_data <- comprehension_all_data %>% 
  select(model, verb_type, rc_type, intended, annotated_answer, item_id) %>% 
  filter(annotated_answer != "else")

comprehension_means <- comprehension_data %>% 
  mutate(intended_numerical = ifelse(intended=="no", 0, 1),
         annotated_numerical = ifelse(annotated_answer=="no", 0, 1)) %>% 
  group_by(model, verb_type, rc_type) %>% 
  summarize(Mean = mean(annotated_numerical),
            CILow = ci.low(annotated_numerical),
            CIHigh = ci.high(annotated_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

# 2. Plot ----
comprehension_graph <- ggplot(comprehension_means,
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
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       # guide="none",
                       name="RC Type") +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                position=position_dodge(width=0.8),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of explanation answers",
       x = "Verb Type") +
  facet_grid(. ~ model) +
  theme(legend.position = "top",
        strip.text.x = element_text(size=12),
        legend.text = element_text(size=12),
        legend.title = element_text(size=12),
        axis.text.x = element_text(size = 10),
        axis.title.x = element_text(size=12),
        axis.text.y = element_text(size = 10),
        axis.title.y = element_text(size=12)) +
  ylim(0,1) +
  scale_fill_brewer(palette = "Dark2", guide="none")
comprehension_graph
ggsave(comprehension_graph, file="../graphs/comprehension_nonIC_why.pdf", width=8, height=5)

# 3. Statistical analysis ----
## gpt-3.5-turbo ----
comprehension.gpt35.data <- comprehension.gpt35.data %>% 
  mutate(verb_type = fct_relevel(as.factor(verb_type), ref="IC"),
         rc_type = fct_relevel(as.factor(rc_type), ref="exp"),
         annotated_answer = as.factor(annotated_answer))
# levels(comprehension.gpt35.data$verb_type)
contrasts(comprehension.gpt35.data$rc_type) = contr.sum(2)
contrasts(comprehension.gpt35.data$verb_type) = contr.sum(2)
gpt35 <- glmer(annotated_answer ~ verb_type * rc_type + (1+verb_type|item_id),
               family = "binomial",
               # fixef.prior = normal(cov = diag(9,4)),
               data=comprehension.gpt35.data)
summary(gpt35)

gpt35_why_bayesian_default_prior_dummy <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                                        family = "bernoulli",
                                              data=comprehension.gpt35.data,
                                              iter=8000,
                                              warmup = 4000,
                                              chains=4,
                                              cores=4,
                                              control=list(max_treedepth = 15, adapt_delta = 0.99),
                                              file="../cache/gpt35_why_bayesian_default_prior_dummy",
                                              seed=1024)
summary(gpt35_why_bayesian_default_prior_dummy)

conditional_effects(gpt35_why_bayesian_default_prior_dummy)
fixef_data<-data.frame(fixef(gpt35_why_bayesian_default_prior_dummy, summary = F))
fixef_data %>%
  rename(interaction="verb_typenonIC.rc_typenonexp") %>%
  summarize(sum_IC = sum(verb_typenonIC<0),
            proportion_IC = sum_IC/n(),
            sum_interaction = sum(interaction>0),
            proportion_interaction = sum_interaction/n())

gpt35_emms <- emmeans(gpt35_why_bayesian_default_prior_dummy,specs = ~ rc_type | verb_type)
pairs(gpt35_emms)

## gpt-4 ----
comprehension.gpt4.data <- comprehension.gpt4.data %>% 
  mutate(verb_type = fct_relevel(as.factor(verb_type), ref="IC"),
         rc_type = fct_relevel(as.factor(rc_type), ref="exp"),
         annotated_answer = as.factor(annotated_answer))
# contrasts(comprehension.gpt4.data$rc_type) = contr.sum(2)
# contrasts(comprehension.gpt4.data$verb_type) = contr.sum(2)
gpt4 <- glmer(annotated_answer ~ verb_type * rc_type + (0+verb_type|item_id),
               family = "binomial",
               # fixef.prior = normal(cov = diag(9,4)),
               data=comprehension.gpt4.data)
summary(gpt4)

gpt4_why_bayesian_default_prior_dummy <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                                              family = "bernoulli",
                                              data=comprehension.gpt4.data,
                                              iter=8000,
                                              warmup = 4000,
                                              chains=4,
                                              cores=4,
                                              control=list(max_treedepth = 15, adapt_delta = 0.99),
                                              file="../cache/gpt4_why_bayesian_default_prior_dummy",
                                              seed=1024)
summary(gpt4_why_bayesian_default_prior_dummy)

## gpt-4o ----
comprehension.gpt4o.data <- comprehension.gpt4o.data %>% 
  mutate(verb_type = fct_relevel(as.factor(verb_type), ref="IC"),
         rc_type = fct_relevel(as.factor(rc_type), ref="exp"),
         annotated_answer = as.factor(annotated_answer))
# levels(comprehension.gpt35.data$verb_type)
contrasts(comprehension.gpt4o.data$rc_type) = contr.sum(2)
contrasts(comprehension.gpt4o.data$verb_type) = contr.sum(2)
gpt4o <- glmer(annotated_answer ~ verb_type * rc_type + (0+verb_type|item_id),
              family = "binomial",
              # fixef.prior = normal(cov = diag(9,4)),
              data=comprehension.gpt4o.data)
summary(gpt4o)

gpt4o_why_bayesian_default_prior_dummy <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                                             family = "bernoulli",
                                             data=comprehension.gpt4o.data,
                                             iter=8000,
                                             warmup = 4000,
                                             chains=4,
                                             cores=4,
                                             control=list(max_treedepth = 15, adapt_delta = 0.99),
                                             file="../cache/gpt4o_why_bayesian_default_prior_dummy",
                                             seed=1024)
summary(gpt4o_why_bayesian_default_prior_dummy)

## Llama-3.2-1B-Instruct ----
comprehension.llama1B_instruct.data <- comprehension.llama1B_instruct.data %>% 
  mutate(verb_type = fct_relevel(as.factor(verb_type), ref="IC"),
         rc_type = fct_relevel(as.factor(rc_type), ref="exp"),
         annotated_answer = as.factor(annotated_answer))
# contrasts(comprehension.llama1B_instruct.data$rc_type) = contr.sum(2)
# contrasts(comprehension.llama1B_instruct.data$verb_type) = contr.sum(2)
llama1B_instruct <- glmer(annotated_answer ~ verb_type * rc_type + (0+verb_type|item_id),
               family = "binomial",
               # fixef.prior = normal(cov = diag(9,4)),
               data=comprehension.llama1B_instruct.data)
summary(llama1B_instruct)

llama1B_instruct_why_bayesian_default_prior_dummy <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                                             family = "bernoulli",
                                             data=comprehension.llama1B_instruct.data,
                                             iter=8000,
                                             warmup = 4000,
                                             chains=4,
                                             cores=4,
                                             control=list(max_treedepth = 15, adapt_delta = 0.99),
                                             file="../cache/llama1B_instruct_why_bayesian_default_prior_dummy",
                                             seed=1024)
summary(llama1B_instruct_why_bayesian_default_prior_dummy)

conditional_effects(llama1B_instruct_why_bayesian_default_prior_dummy)
fixef_data<-data.frame(fixef(llama1B_instruct_why_bayesian_default_prior_dummy, summary = F))
fixef_data %>%
  rename(interaction="verb_typenonIC.rc_typenonexp") %>%
  summarize(sum_RC = sum(rc_typenonexp<0),
            proportion_IC = sum_RC/n(),
            sum_interaction = sum(interaction>0),
            proportion_interaction = sum_interaction/n())

llama1B_instruct_emms <- emmeans(llama1B_instruct_why_bayesian_default_prior_dummy,specs = ~ rc_type | verb_type)
pairs(llama1B_instruct_emms)

## Llama-3.2-3B-Instruct ----
comprehension.llama3B_instruct.data <- comprehension.llama3B_instruct.data %>% 
  mutate(verb_type = fct_relevel(as.factor(verb_type), ref="IC"),
         rc_type = fct_relevel(as.factor(rc_type), ref="exp"),
         annotated_answer = as.factor(annotated_answer))
# contrasts(comprehension.llama3B_instruct.data$rc_type) = contr.sum(2)
# contrasts(comprehension.llama3B_instruct.data$verb_type) = contr.sum(2)
llama3B_instruct <- glmer(annotated_answer ~ verb_type * rc_type + (1+verb_type|item_id),
                          family = "binomial",
                          # fixef.prior = normal(cov = diag(9,4)),
                          data=comprehension.llama3B_instruct.data)
summary(llama3B_instruct)

llama3B_instruct_why_bayesian_default_prior_dummy <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                                                         family = "bernoulli",
                                                         data=comprehension.llama3B_instruct.data,
                                                         iter=8000,
                                                         warmup = 4000,
                                                         chains=4,
                                                         cores=4,
                                                         control=list(max_treedepth = 15, adapt_delta = 0.99),
                                                         file="../cache/llama3B_instruct_why_bayesian_default_prior_dummy",
                                                         seed=1024)
summary(llama3B_instruct_why_bayesian_default_prior_dummy)
llama3B_emms <- emmeans(llama3B_instruct_why_bayesian_default_prior_dummy,specs = ~ rc_type | verb_type)
pairs(llama3B_emms)

conditional_effects(llama3B_instruct_why_bayesian_default_prior_dummy)
fixef_data<-data.frame(fixef(llama3B_instruct_why_bayesian_default_prior_dummy, summary = F))
fixef_data %>%
  rename(interaction="verb_typenonIC.rc_typenonexp") %>%
  summarize(sum_IC = sum(verb_typenonIC<0),
            proportion_IC = sum_IC/n(),
            sum_interaction = sum(interaction>0),
            proportion_interaction = sum_interaction/n())


# 4. Exploratory type analysis ----
comprehension_response_type <- comprehension_all_data %>% 
  mutate(error_type = if_else(intended=="yes", "intended_infer", error_type)) %>% 
  mutate(error_type = if_else(annotated_answer == "else", "others", error_type)) %>%
  mutate(error_type = if_else(grepl("extended inference", error_type), "extended inference", error_type)) %>% 
  pivot_wider(names_from = error_type, values_from = annotated_answer) %>% 
  rename(extended = "extended inference",
         rc = "rc as reason",
         idk = "idk in ic-exp condition") %>% 
  mutate(intended_infer = if_else(is.na(intended_infer), 0, 1),
         extended = if_else(is.na(extended), 0, 1),
         rc = if_else(is.na(rc), 0, 1),
         idk = if_else(is.na(idk), 0, 1),
         others = if_else(is.na(others), 0, 1))

comprehension_response_type <- comprehension_response_type %>% 
  pivot_longer(cols=c(intended_infer, extended, rc, idk, others),
               values_to = "error_exist",
               names_to = "error_type")

comprehension_response_type_mean <- comprehension_response_type %>%
  group_by(error_type, rc_type, verb_type, model) %>%
  summarize(Mean = mean(error_exist),
            CILow = ci.low(error_exist),
            CIHigh = ci.high(error_exist)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,
         YMax=Mean+CIHigh) %>% 
  mutate(error_type = if_else(error_type == "intended_infer", "intended", error_type),
         condition = paste(verb_type, "+", rc_type)) %>% 
  mutate(error_type = factor(error_type, levels=c("intended", "rc", "extended", "idk", "others"))) 
model_names <- list(
  "llama1B_instruct" = "1B_instruct",
  "llama3B_instruct" = "3B_instruct",
  "gpt-3.5-turbo" = "3.5-turbo",
  "gpt-4" = "gpt-4",
  "gpt-4o" = "gpt-4o"
)

model_labeller <- function(variable,value){
  return(model_names[variable])
}

error_type_names <- list(
  "intended" = "intended\ninference",
  "rc" = "rc content",
  "extended" = "extended\ninference",
  "idk" = "I don't know",
  "others" = "others"
)

error_type_labeller <- function(variable,value){
  return(error_type_names[variable])
}

answer_type_graph <- ggplot(comprehension_response_type_mean,
                            aes(x=condition, y=Mean,
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
  scale_x_discrete(labels = NULL, breaks=NULL) +
  labs(y = "Proprotion of responses",
       x = "") +
  facet_grid(model~error_type, labeller=labeller(model=model_labeller,
                                                 error_type = error_type_labeller)) +
  theme(legend.position = "top",
        strip.text.x = element_text(size = 12),
        strip.text.y = element_text(size = 12),
        legend.text = element_text(size=12),
        axis.text.x = element_text(angle = 60, vjust = 0.5, hjust=0.5, size = 12),
        axis.title.x = element_text(size = 14),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size = 14))
answer_type_graph
ggsave(answer_type_graph, file="../graphs/comprehension_nonIC_full_why_answer_type.pdf", width=8, height=6)
