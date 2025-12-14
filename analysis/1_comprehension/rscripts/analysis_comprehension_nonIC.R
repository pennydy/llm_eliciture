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
library(bridgesampling)


theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
## gpt-4o ----
# adding a column annotated_response after checking whether the "yes" response is followed 
# by an actual explanation
# gpt-3.5-turbo has one response with "Yes,this sentence does not explain why Bob greeted the leader." 
# so it is coded as "no" in annotated_response column
comprehension_alt.gpt4o.data <- read.csv("../../../data/comprehension_nonIC_full_alt/comprehension_nonIC_full_alt-gpt-4o_1_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no")) %>% 
  rename(rc_type = "continuation_type")

# add one
# comprehension_alt.gpt4o.data <- comprehension_alt.gpt4o.data %>% 
#   add_row(sent_id=241,item_id=61,verb="smooth",verb_type="IC",rc_type="exp",explanation="yes", "annotated_answer"="yes") %>% 
#   add_row(sent_id=242,item_id=61,verb="smooth",verb_type="nonIC",rc_type="exp",explanation="yes", "annotated_answer"="yes") %>% 
#   add_row(sent_id=243,item_id=61,verb="smooth",verb_type="IC",rc_type="nonexp",explanation="yes", "annotated_answer"="yes") %>% 
#   add_row(sent_id=244,item_id=61,verb="smooth",verb_type="nonIC",rc_type="nonexp",explanation="yes", "annotated_answer"="yes")

## gpt-4 ----
comprehension_alt.gpt4.data <- read.csv("../../../data/comprehension_nonIC_full_alt/comprehension_nonIC_full_alt-gpt-4_1_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no")) %>% 
  rename(rc_type = "continuation_type")

## gpt-3.5-turbo ----
comprehension_alt.gpt35.data <- read.csv("../../../data/comprehension_nonIC_full_alt/comprehension_nonIC_full_alt-gpt-3.5-turbo_1_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no")) %>% 
  rename(rc_type = "continuation_type")

## Llama-3.2-1B-Instruct ----
comprehension_alt.1B.instruct.data <- read.csv("../../../data/comprehension_nonIC_full_alt/comprehension_full_alt_Llama-3.2-1B-Instruct_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no")) %>% 
  rename(rc_type = "continuation_type")

## Llama-3.2-3B-Instruct ----
comprehension_alt.3B.instruct.data <- read.csv("../../../data/comprehension_nonIC_full_alt/comprehension_full_alt_Llama-3.2-3B-Instruct_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no")) %>% 
  rename(rc_type = "continuation_type")

comprehension_alt_data <- bind_rows(lst(comprehension_alt.gpt4o.data, comprehension_alt.gpt4.data, comprehension_alt.gpt35.data,comprehension_alt.1B.instruct.data,comprehension_alt.3B.instruct.data), .id="model") %>% 
  mutate(model = case_when(model == "comprehension_alt.gpt4o.data" ~ "gpt-4o",
                           model == "comprehension_alt.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "comprehension_alt.gpt4.data" ~ "gpt-4",
                           model == "comprehension_alt.1B.instruct.data" ~ "1B-Instruct",
                           model == "comprehension_alt.3B.instruct.data" ~ "3B-Instruct"),
         explanation = if_else(explanation == "no", "no", "yes")) %>% 
  select(model, verb_type, rc_type, explanation, annotated_answer, item_id)

comprehension_alt_means <- comprehension_alt_data %>% 
  mutate(explanation_numerical = ifelse(annotated_answer=="no", 0, 1)) %>% 
  group_by(model, verb_type, rc_type) %>% 
  summarize(Mean = mean(explanation_numerical),
            CILow = ci.low(explanation_numerical),
            CIHigh = ci.high(explanation_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

# 2. Plot ----
annotation_df <- data.frame(
  model = c("gpt-3.5-turbo", "gpt-4", "gpt-4o"),
  start = "IC",
  end = "nonIC",
  y=c(1, 0.9,0.96),
  label = c("***", "***", "***")
)

comprehension_alt_graph <- ggplot(comprehension_alt_means,
                                  aes(x=verb_type,y=Mean,
                                      fill=verb_type,
                                      pattern=rc_type)) +
  geom_bar_pattern(
    position = "dodge",
    stat="identity",
    pattern_angle = 45,
    pattern_spacing = 0.1,
    pattern_fill="black",
    pattern_alpha=0.6,
    alpha=0.7) +
  # geom_bar(stat="identity",
  #          position="dodge2",
  #          aes(fill=verb_type,
  #              alpha=continuation_type),
  #          width=0.8)+
  scale_pattern_manual(values=c(exp = "stripe", nonexp = "none"),
                       name="RC Type") +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),
                position=position_dodge(width=0.8),
                width=.2, 
                show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of explanation answers",
       x = "Verb Type") +
  # geom_signif(data=annotation_df,
  #             aes(y_position=y,
  #                 xmin = start,
  #                 xmax = end,
  #                 annotations=label),
  #             textsize = 4,
  #             manual=TRUE) +
  facet_grid(. ~ model) +
  theme(legend.position = "top",
        strip.text.x = element_text(size=9),
        legend.text = element_text(size=14),
        legend.title = element_text(size=12),
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size=14),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(size=14)) +
  ylim(0,1) +
  scale_fill_brewer(palette = "Dark2", guide="none")
comprehension_alt_graph
ggsave(comprehension_alt_graph, file="../graphs/comprehension_nonIC_full_all.pdf", width=7, height=4)

# 3. Statistical analysis ----
## gpt-3.5-turbo ----
# options(contrasts = c("contr.sum","contr.sum"))
comprehension_alt.gpt35.data <- comprehension_alt.gpt35.data %>% 
  mutate(verb_type = fct_relevel(as.factor(verb_type), "IC"),
         rc_type = fct_relevel(as.factor(rc_type),"exp"),
         annotated_answer = as.factor(annotated_answer))
# levels(comprehension_alt.gpt35.data$rc_type)
# gpt35 <- glmer(annotated_answer ~ verb_type * continuation_type + (1|sent_id),
#                family = "binomial",
#                data=comprehension_alt.gpt35.data)
gpt35 <- glmer(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
               family = "binomial",
               data=comprehension_alt.gpt35.data)
summary(gpt35)

gpt35_bayesian_default_prior <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                     family="bernoulli",
                     data=comprehension_alt.gpt35.data,
                     iter=8000,
                     warmup = 4000,
                     chains=4,
                     cores=4,
                     control=list(max_treedepth = 15, adapt_delta = 0.99),
                     file="../cache/brm_gpt35_default_prior",
                     seed = 1024)
gpt35_bayesian_default_prior <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                                    family="bernoulli",
                                    data=comprehension_alt.gpt35.data,
                                    iter=8000,
                                    warmup = 4000,
                                    chains=4,
                                    cores=4,
                                    control=list(max_treedepth = 15, adapt_delta = 0.99),
                                    file="../cache/brm_gpt35_default_prior_dummy",
                                    seed = 1024)

summary(gpt35_bayesian_default_prior)


gpt35_bayesian <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                      family="bernoulli",
                      data=comprehension_alt.gpt35.data,
                      iter=8000,
                      warmup = 4000,
                      chains=4,
                      cores=4,
                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                      prior = c(set_prior("normal(0,3)", class = "b")),
                      # file="../cache/brm_gpt35",
                      seed = 1024)
summary(gpt35_bayesian)

## gpt-4 ----
comprehension_alt.gpt4.data <- comprehension_alt.gpt4.data %>% 
  mutate(verb_type = fct_relevel(as.factor(verb_type),"IC"),
         rc_type = fct_relevel(as.factor(rc_type),"exp"),
         annotated_answer = as.factor(annotated_answer))
contrasts(comprehension_alt.gpt4.data$verb_type)=contr.sum(2)
contrasts(comprehension_alt.gpt4.data$rc_type)=contr.sum(2)

# there is no explanation response in the IC condition, so there're warnings
# about not positive definite or contains NA value
# gpt4 <- glmer(annotated_answer ~ verb_type * continuation_type + (1|sent_id),
#               family = "binomial",
#               data=comprehension_alt.gpt4.data)

# gpt4.data <- comprehension_alt.gpt4.data %>% 
#   select(-c("main_clause", "target", "question", "prompt", "answer", "raw_probs"))
# write.csv(gpt4.data, "./gpt4_data.csv")
# comprehension_alt.gpt4.data$verb_type <- factor(comprehension_alt.gpt4.data$verb_type, levels = c("nonIC", "IC"))
# comprehension_alt.gpt4.data$rc_type <- factor(comprehension_alt.gpt4.data$rc_type, levels = c("nonexp", "exp"))
gpt4 <- glmer(annotated_answer ~ verb_type + rc_type + (1+rc_type + verb_type|item_id),
              family = "binomial",
              control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e6)),
              data=comprehension_alt.gpt4.data)
summary(gpt4)

gpt4_brm <- brm(
  annotated_answer ~ verb_type * rc_type + (1+rc_type+verb_type| item_id),
  data = comprehension_alt.gpt4.data,
  family = bernoulli(),
  chains = 4, cores = 4, iter = 2000,
  control = list(adapt_delta = 0.95),
  seed=1024
)
summary(gpt4_brm)


gpt4_bayesian_default_prior <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                     family="bernoulli",
                     data=comprehension_alt.gpt4.data,
                     iter=8000,
                     warmup = 4000,
                     chains=4,
                     cores=4,
                     control=list(max_treedepth = 15, adapt_delta = 0.99),
                     file="../cache/brm_gpt4_default_prior",
                     seed = 1024)

gpt4_bayesian_default_prior <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                                   family="bernoulli",
                                   data=comprehension_alt.gpt4.data,
                                   iter=8000,
                                   warmup = 4000,
                                   chains=4,
                                   cores=4,
                                   control=list(max_treedepth = 15, adapt_delta = 0.99),
                                   file="../cache/brm_gpt4_default_prior_dummy",
                                   seed = 1024)
summary(gpt4_bayesian_default_prior)
prior_summary(gpt4_bayesian_default_prior)

fixef_data<-data.frame(fixef(gpt4_bayesian_default_prior, summary = F))
fixef_data %>%
  rename(interaction="verb_type1.rc_type1") %>%
  summarize(sum = sum(rc_type1>0),
            proportion = sum/n()) 
 
 
gpt4_bayesian <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                                    family="bernoulli",
                                    data=comprehension_alt.gpt4.data,
                                    iter=8000,
                                    warmup = 4000,
                                    chains=4,
                                    cores=4,
                                    control=list(max_treedepth = 15, adapt_delta = 0.99),
                                    # prior = c(set_prior("student_t(3, 0, 3)", class = "b"),
                                    #           set_prior("student_t(3, 0, 3)", class = "sd")),
                                    prior=c(set_prior("normal(0,3)", class="b")),
                     # file="../cache/brm_gpt4",
                     seed = 1024)

gpt4_bayesian <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                     family="bernoulli",
                     data=comprehension_alt.gpt4.data,
                     iter=8000,
                     warmup = 4000,
                     chains=4,
                     cores=4,
                     control=list(max_treedepth = 15, adapt_delta = 0.99),
                     prior = c(set_prior("student_t(3, 0, 3)", class = "b"),
                               set_prior("student_t(3, 0, 3)", class = "sd")),
                     file="../cache/brm_gpt4_dummy",
                     seed = 1024)

summary(gpt4_bayesian)
 
conditional_effects(gpt4_bayesian)
fixef_data<-data.frame(fixef(gpt4_bayesian, summary = F))
fixef_data %>%
  rename(interaction="verb_type1.rc_type1") %>%
  summarize(sum = sum(rc_type1>0),
            proportion = sum/n())

gpt4_bayesian_prior_10 <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                     family="bernoulli",
                     data=comprehension_alt.gpt4.data,
                     iter=8000,
                     warmup = 4000,
                     chains=4,
                     cores=4,
                     control=list(max_treedepth = 15, adapt_delta = 0.99),
                     prior = c(set_prior("student_t(3, 0, 10)", class = "b"),
                               set_prior("student_t(3, 0, 10)", class = "sd")),
                     seed = 1024)
summary(gpt4_bayesian_prior_10)
conditional_effects(gpt4_bayesian_prior_10)

hypothesis(gpt4_bayesian, c('rc_type1 + verb_type1:rc_type1 = 0', 
                                      'rc_type1 - verb_type1:rc_type1 = 0'))

fixef_data<-data.frame(fixef(gpt4_bayesian_prior_10, summary = F))
fixef_data %>%
  rename(interaction="verb_type1.rc_type1") %>%
  summarize(sum = sum("rc_type1">0),
            proportion = sum/n())

## gpt-4o ----
comprehension_alt.gpt4o.data <- comprehension_alt.gpt4o.data %>%
  mutate(verb_type = fct_relevel(as.factor(verb_type),"IC"),
         rc_type = fct_relevel(as.factor(rc_type),"exp"),
         annotated_answer = as.factor(annotated_answer))

gpt4o.data <- comprehension_alt.gpt4o.data %>%
  select(-c("main_clause", "target", "question", "prompt", "answer", "raw_probs"))
write.csv(gpt4o.data, "./gpt4o_data.csv")

# comprehension_alt.gpt4o.data_sum <- comprehension_alt.gpt4o.data_sum %>% 
#   mutate(verb_type = if_else(verb_type=="IC", 1, 0),
#          rc_type = if_else(rc_type=="exp",1,0))
# 
# comprehension_alt.gpt4o.data_sum$verb_type <- scale(comprehension_alt.gpt4o.data_sum$verb_type, center = TRUE, scale = FALSE)
# comprehension_alt.gpt4o.data_sum$rc_type <- scale(comprehension_alt.gpt4o.data_sum$rc_type, center = TRUE, scale = FALSE)

# contrasts(comprehension_alt.gpt4o.data_sum$verb_type)=contr.sum(2)
# contrasts(comprehension_alt.gpt4o.data_sum$rc_type)=contr.sum(2)

# gpt4o <- glmer(annotated_answer ~ verb_type * rc_type + (1|sent_id),
#                family = "binomial",
#                data=comprehension_alt.gpt4o.data)
# gpt4o <- glmer(annotated_answer ~ verb_type * rc_type + (1|sent_id),
#                family = "binomial",
#                data=comprehension_alt.gpt4o.data)
gpt4o <- glmer(annotated_answer ~ verb_type * rc_type + (1|item_id),
               family = "binomial",
               control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)),
               data=comprehension_alt.gpt4o.data)
summary(gpt4o)

gpt4o_brm <- brm(
  annotated_answer ~ verb_type * rc_type + (1+rc_type+verb_type| item_id),
  data = comprehension_alt.gpt4o.data,
  family = bernoulli(),
  save_pars = save_pars(all = TRUE), 
  chains = 4, cores = 4, iter = 2000,
  control = list(adapt_delta = 0.95),
  seed=1024
)
summary(gpt4o_brm)
conditional_effects(gpt4o_brm)
fixef_data<-data.frame(fixef(gpt4o_brm, summary = F))
fixef_data %>%
  rename(interaction="verb_typenonIC.rc_typenonexp") %>%
  summarize(sum = sum(interaction<0),
            proportion = sum/n())

gpt4o_brm_simple <- brm(
  annotated_answer ~ verb_type + rc_type + (1+rc_type+verb_type| item_id),
  data = comprehension_alt.gpt4o.data,
  family = bernoulli(),
  save_pars = save_pars(all = TRUE), 
  chains = 4, cores = 4, iter = 2000,
  control = list(adapt_delta = 0.95),
  seed=1024
)
summary(gpt4o_brm_simple)

bf_inter <- bridge_sampler(gpt4o_brm, recompile=TRUE)
bf_nointer <- bridge_sampler(gpt4o_brm_simple, recompile=TRUE)
bayes_factor(bf_inter, bf_nointer)

gpt4o_bayesian_default_prior <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+ rc_type|item_id),
                     family="bernoulli",
                     data=comprehension_alt.gpt4o.data,
                     iter=8000,
                     warmup = 4000,
                     chains=4,
                     cores=4,
                     control=list(max_treedepth = 15, adapt_delta = 0.99),
                     file="../cache/brm_gpt4o_default_prior",
                     seed=1024)

gpt4o_bayesian_default_prior <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+rc_type|item_id),
                                    family="bernoulli",
                                    data=comprehension_alt.gpt4o.data,
                                    iter=8000,
                                    warmup = 4000,
                                    chains=4,
                                    cores=4,
                                    save_pars = save_pars(all = TRUE), 
                                    control=list(max_treedepth = 15, adapt_delta = 0.99),
                                    file="../cache/brm_gpt4o_default_prior_dummy",
                                    seed=1024)
summary(gpt4o_bayesian_default_prior)
prior_summary(gpt4o_bayesian_default_prior)

gpt4o_bayesian_default_prior_simple <- brm(annotated_answer ~ verb_type + rc_type + (1+verb_type+rc_type|item_id),
                                    family="bernoulli",
                                    data=comprehension_alt.gpt4o.data,
                                    iter=8000,
                                    warmup = 4000,
                                    chains=4,
                                    cores=4,
                                    save_pars = save_pars(all = TRUE), 
                                    control=list(max_treedepth = 15, adapt_delta = 0.99),
                                    file="../cache/brm_gpt4o_default_prior_dummy_simple",
                                    seed=1024)
summary(gpt4o_bayesian_default_prior_simple)

bf_inter <- bridge_sampler(gpt4o_bayesian_default_prior, recompile=TRUE)
bf_nointer <- bridge_sampler(gpt4o_bayesian_default_prior_simple, recompile=TRUE)
bayes_factor(bf_inter, bf_nointer)

conditional_effects(gpt4o_bayesian_default_prior)
fixef_data<-data.frame(fixef(gpt4o_bayesian_default_prior, summary = F))
fixef_data %>%
  rename(interaction="verb_typenonIC.rc_typenonexp") %>%
  summarize(sum = sum(interaction<0),
            proportion = sum/n())

hypothesis(gpt4o_bayesian_default_prior,c('rc_typenonexp + verb_typenonIC:rc_typenonexp = 0', 
             'rc_typenonexp - verb_typenonIC:rc_typenonexp = 0'))

# use a different prior
gpt4o_bayesian <- brm(annotated_answer ~ verb_type * rc_type + (1|item_id),
                      family="bernoulli",
                      data=comprehension_alt.gpt4o.data,
                      iter=8000,
                      warmup = 4000,
                      chains=4,
                      cores=4,
                      control=list(max_treedepth = 15, adapt_delta = 0.99),
                      prior = c(set_prior("normal(0,3)", class = "b")),
                      # file="../cache/brm_gpt4o_dummy",
                      seed=1024)
summary(gpt4o_bayesian)
prior_summary(gpt4o_bayesian)

conditional_effects(gpt4o_bayesian)
fixef_data<-data.frame(fixef(gpt4o_bayesian, summary = F))
colnames(fixef_data)
fixef_data %>%
  rename(interaction="verb_typenonIC.rc_typenonexp") %>%
  summarize(sum = sum(interaction>0),
            proportion = sum/n())

get_prior(gpt4o_bayesian_mean <- brm(annotated_answer ~ verb_type * rc_type + (1+verb_type+ rc_type|item_id),
                                     family="bernoulli",
                                     data=comprehension_alt.gpt4o.data,
                                     iter=8000,
                                     warmup = 4000,
                                     chains=4,
                                     cores=4,
                                     control=list(max_treedepth = 15, adapt_delta = 0.99))) 
prior_summary(gpt4o_bayesian_mean)
