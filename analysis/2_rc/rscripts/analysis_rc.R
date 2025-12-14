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
library(brms)

theme_set(theme_bw())
# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("helpers.R")

# 1. Data ----
rc_alt.gpt4o.data <- read.csv("../../../data/rc_full_alt/rc_full_alt-gpt-4o_6.csv", header=TRUE) %>% 
  na.omit() 
rc_alt.gpt4o.data <- rc_alt.gpt4o.data %>% 
  mutate(answer_choice = as.numeric(gsub("\\D", "", answer)),
         answer_option = strsplit(answer_option, ", "),
         option_1 = gsub("[^a-zA-Z]", "", lapply(answer_option, `[[`, 1)),
         option_2 = gsub("[^a-zA-Z]", "",lapply(answer_option, `[[`, 2)),
         answer = ifelse(answer_choice == 1, option_1, option_2),
         sg_pl = if_else(answer %in% c("have", "were", "are"), "pl", "sg"),
         attachment = case_when(np_order == "high_sg" & sg_pl == "sg" ~ "high",
                                np_order == "high_sg" & sg_pl == "pl" ~ "low",
                                np_order == "high_pl" & sg_pl == "pl" ~ "high",
                                np_order == "high_pl" & sg_pl == "sg" ~ "low"))

rc_alt.gpt4.data <- read.csv("../../../data/rc_full_alt/rc_full_alt-gpt-4_6.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(answer=as.character(answer))
rc_alt.gpt4.data <- rc_alt.gpt4.data %>% 
  mutate(answer_choice = as.numeric(gsub("\\D", "", answer)),
         answer_option = strsplit(answer_option, ", "),
         option_1 = gsub("[^a-zA-Z]", "", lapply(answer_option, `[[`, 1)),
         option_2 = gsub("[^a-zA-Z]", "",lapply(answer_option, `[[`, 2)),
         answer = ifelse(answer_choice == 1, option_1, option_2),
         sg_pl = if_else(answer %in% c("have", "were", "are"), "pl", "sg"),
         attachment = case_when(np_order == "high_sg" & sg_pl == "sg" ~ "high",
                                np_order == "high_sg" & sg_pl == "pl" ~ "low",
                                np_order == "high_pl" & sg_pl == "pl" ~ "high",
                                np_order == "high_pl" & sg_pl == "sg" ~ "low"))

rc_alt.gpt35.data <- read.csv("../../../data/rc_full_alt/rc_full_alt-gpt-3.5-turbo_6.csv", header=TRUE) %>% 
  na.omit()
rc_alt.gpt35.data <- rc_alt.gpt35.data %>% 
  mutate(answer_choice = as.numeric(gsub("\\D", "", answer)),
         answer_option = strsplit(answer_option, ", "),
         option_1 = gsub("[^a-zA-Z]", "", lapply(answer_option, `[[`, 1)),
         option_2 = gsub("[^a-zA-Z]", "",lapply(answer_option, `[[`, 2)),
         answer = ifelse(answer_choice == 1, option_1, option_2),
         sg_pl = if_else(answer %in% c("have", "were", "are"), "pl", "sg"),
         attachment = case_when(np_order == "high_sg" & sg_pl == "sg" ~ "high",
                                np_order == "high_sg" & sg_pl == "pl" ~ "low",
                                np_order == "high_pl" & sg_pl == "pl" ~ "high",
                                np_order == "high_pl" & sg_pl == "sg" ~ "low"))

rc_alt_data <- bind_rows(lst(rc_alt.gpt4o.data, rc_alt.gpt4.data, rc_alt.gpt35.data), .id="model") %>% 
  mutate(model = case_when(model == "rc_alt.gpt4o.data" ~ "gpt-4o",
                           model == "rc_alt.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "rc_alt.gpt4.data" ~ "gpt-4")) %>% 
         # np_order = if_else(item_id < 11, "high_sg", "high_pl")) %>% 
  select(model, item_id,sentence_type,attachment)

rc_alt_means <- rc_alt_data %>% 
  mutate(attachment_numerical = ifelse(attachment=="high", 1, 0)) %>%
  group_by(model, sentence_type) %>% 
  summarize(Mean = mean(attachment_numerical),
            CILow = ci.low(attachment_numerical),
            CIHigh = ci.high(attachment_numerical)) %>% 
  ungroup() %>% 
  mutate(YMin = Mean-CILow,
         YMax = Mean+CIHigh)

# 2. Plot ----
annotation_df <- data.frame(
  model = c("gpt-3.5-turbo", "gpt-4", "gpt-4o"),
  start = "IC",
  end = "nonIC",
  y=c(0.8, 0.65,0.75),
  label = c("NS", "*", "NS")
)
rc_alt_graph <- ggplot(rc_alt_means,
                       aes(x=sentence_type,y=Mean)) +
  geom_bar(stat="identity", 
           aes(fill=sentence_type),
           width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  geom_signif(data=annotation_df,
              aes(y_position=y,
                  xmin = start,
                  xmax = end,
                  annotations=label),
              textsize = 4,
              manual=TRUE)+
  # geom_signif(comparisons=list(c("IC", "nonIC")), annotations="**",y_position = 0.9) +
  labs(y = "Proportion of high attachment",
       x = "Verb Type") +
  facet_wrap(. ~ model) +
  scale_fill_brewer(palette = "Dark2") +
  theme(legend.position="none",
        strip.text.x = element_text(size = 12),
        axis.title.x = element_text(size=14),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size=14),
        axis.text.y = element_text(size = 12)) +
  ylim(0,1)
rc_alt_graph
ggsave(rc_alt_graph, file="../graphs/rc_alt_6.pdf", width=7, height=4)


# 3. Statistical analysis ----
## gpt-3.5-turbo ----
rc_alt.gpt35.data <- rc_alt.gpt35.data %>% 
  mutate(sentence_type = fct_relevel(as.factor(sentence_type)),
         attachment = as.factor(attachment)) %>% 
  rename(verb_type="sentence_type")
gpt35 <- glmer(attachment ~ verb_type + (1|item_id),
               family = "binomial",
               data=rc_alt.gpt35.data)
summary(gpt35)

gpt35_bayesian_default_prior <- brm(attachment ~ verb_type + (1|item_id),
                                    family="bernoulli",
                                   data=rc_alt.gpt35.data,
                                   iter=8000,
                                   warmup = 4000,
                                   chains=4,
                                   cores=4,
                                   control=list(max_treedepth = 15, adapt_delta = 0.99),
                                   file="../cache/gpt35_default_prior",
                                   seed=1024)
summary(gpt35_bayesian_default_prior)

## gpt-4 ----
rc_alt.gpt4.data <- rc_alt.gpt4.data %>% 
  mutate(sentence_type = fct_relevel(as.factor(sentence_type)),
         attachment = as.factor(attachment)) %>% 
  rename(verb_type = "sentence_type")
gpt4 <- glmer(attachment ~ sentence_type + (1|item_id),
               family = "binomial",
               data=rc_alt.gpt4.data)
summary(gpt4)

gpt4_bayesian_default_prior <- brm(attachment ~ verb_type + (1|item_id),
                                    family="bernoulli",
                                    data=rc_alt.gpt4.data,
                                    iter=8000,
                                    warmup = 4000,
                                    chains=4,
                                    cores=4,
                                    control=list(max_treedepth = 15, adapt_delta = 0.99),
                                    file="../cache/gpt4_default_prior",
                                    seed=1024)
summary(gpt4_bayesian_default_prior)

## gpt-4o ----
rc_alt.gpt4o.data <- rc_alt.gpt4o.data %>% 
  mutate(sentence_type = fct_relevel(as.factor(sentence_type)),
         attachment = as.factor(attachment)) %>% 
  rename(verb_type = "sentence_type")

gpt4o <- glmer(attachment ~ verb_type + (1|item_id),
              family = "binomial",
              data=rc_alt.gpt4o.data)
summary(gpt4o)

gpt4o_bayesian_default_prior <- brm(attachment ~ verb_type + (1|item_id),
                                   family="bernoulli",
                                   data=rc_alt.gpt4o.data,
                                   iter=8000,
                                   warmup = 4000,
                                   chains=4,
                                   cores=4,
                                   control=list(max_treedepth = 15, adapt_delta = 0.99),
                                   file="../cache/gpt4o_default_prior",
                                   seed=1024)
summary(gpt4o_bayesian_default_prior)
