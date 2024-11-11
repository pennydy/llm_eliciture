library(lme4)
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
# adding a column annotated_response after checking whether the "yes" response is followed 
# by an actual explanation
# gpt-3.5-turbo has one response with "Yes,this sentence does not explain why Bob greeted the leader." 
# so it is coded as "no" in annotated_response column
comprehension_alt.gpt4o.data <- read.csv("../../../data/comprehension_full_alt/comprehension_full_alt-gpt-4o_3_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no"))
comprehension_alt.gpt4.data <- read.csv("../../../data/comprehension_full_alt/comprehension_full_alt-gpt-4_3_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no"))
comprehension_alt.gpt35.data <- read.csv("../../../data/comprehension_full_alt/comprehension_full_alt-gpt-3.5-turbo_3_annotate.csv", header=TRUE) %>% 
  na.omit() %>% 
  mutate(explanation = ifelse(grepl("Yes", answer, ignore.case = TRUE), "yes", "no"))

comprehension_alt_data <- bind_rows(lst(comprehension_alt.gpt4o.data, comprehension_alt.gpt4.data, comprehension_alt.gpt35.data), .id="model") %>% 
  mutate(model = case_when(model == "comprehension_alt.gpt4o.data" ~ "gpt-4o",
                           model == "comprehension_alt.gpt35.data" ~ "gpt-3.5-turbo",
                           model == "comprehension_alt.gpt4.data" ~ "gpt-4"),
         explanation = if_else(explanation == "no", "no", "yes")) %>% 
  select(model, sentence_type, explanation, annotated_answer, item_id)

comprehension_alt_means <- comprehension_alt_data %>% 
  mutate(explanation_numerical = ifelse(annotated_answer=="no", 0, 1)) %>% 
  group_by(model, sentence_type) %>% 
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
                                  aes(x=sentence_type,y=Mean)) +
  geom_bar(stat="identity", 
           aes(fill=sentence_type),
           width=0.8, alpha=0.7)+
  geom_errorbar(aes(ymin=YMin,ymax=YMax), width=.2, show.legend = FALSE) +
  theme_bw() +
  labs(y = "Proportion of explanation answers",
       x = "Verb Type") +
  geom_signif(data=annotation_df,
              aes(y_position=y,
                  xmin = start,
                  xmax = end,
                  annotations=label),
              textsize = 4,
              manual=TRUE) +
  facet_wrap(. ~ model) +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10)) +
  ylim(0,1) +
  scale_fill_brewer(palette = "Dark2")
comprehension_alt_graph
ggsave(comprehension_alt_graph, file="../graphs/comprehension_full_alt_3.pdf", width=7, height=4)

# 3. Statistical analysis ----
## gpt-3.5-turbo ----
comprehension_alt.gpt35.data <- comprehension_alt.gpt35.data %>% 
  mutate(sentence_type = fct_relevel(as.factor(sentence_type)),
         annotated_answer = as.factor(annotated_answer))
gpt35 <- glmer(annotated_answer ~ sentence_type + (1|sent_id),
               family = "binomial",
               data=comprehension_alt.gpt35.data)
summary(gpt35)

## gpt-4 ----
comprehension_alt.gpt4.data <- comprehension_alt.gpt4.data %>% 
  mutate(sentence_type = fct_relevel(as.factor(sentence_type)),
         annotated_answer = as.factor(annotated_answer))
# there is no explanation response in the IC condition, so there're warnings
# about not positive definite or contains NA value
gpt4 <- glmer(annotated_answer ~ sentence_type + (1|sent_id),
               family = "binomial",
               data=comprehension_alt.gpt4.data)
summary(gpt4)

## gpt-4o ----
comprehension_alt.gpt4o.data <- comprehension_alt.gpt4o.data %>% 
  mutate(sentence_type = fct_relevel(as.factor(sentence_type)),
         annotated_answer = as.factor(annotated_answer))
gpt4o <- glmer(annotated_answer ~ sentence_type + (1|sent_id),
              family = "binomial",
              data=comprehension_alt.gpt4o.data)
summary(gpt4o)
