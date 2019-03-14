library(tidyverse)
library(tidytext)

trolls_non_trolls <-
  read_csv("trolls_non_trolls2.csv", col_types = "ccc") %>%
  select(-1) %>%
  mutate(label = ifelse(label == 1, "troll", "real_tweet"))

trolls_non_trolls <- trolls_non_trolls[complete.cases(trolls_non_trolls), ] %>%
  filter(!str_detect(tweet_text, "^null$"))

trolls_non_trolls %>% head()

data("stop_words")

tidy_trolls <- trolls_non_trolls %>%
  unnest_tokens(word, tweet_text)

tidy_trolls <- tidy_trolls %>%
  anti_join(stop_words, by = "word")

tidy_trolls_n <- tidy_trolls %>%
  count(label, word, sort = TRUE)


tidy_trolls_ngrams <- trolls_non_trolls %>%
  anti_join(stop_words, by = c("tweet_text" = "word"))

tidy_trolls_ngrams <- tidy_trolls_ngrams %>%
  unnest_tokens(ngram, tweet_text, token = "ngrams", n = 3)

tidy_trolls_ngrams_n <- tidy_trolls_ngrams %>%
  count(label, ngram, sort = TRUE)

library(rlang)
library(stringr)
plot_words <- function(df, ngram, group, title) {
  ngram <- enquo(ngram)
  fill_color <- ifelse(group == "troll", "#2a3990", "darkgrey")
  df %>%
    filter(label == group) %>%
    top_n(20, wt = n) %>%
    mutate(word = reorder(!! ngram, n)) %>%
    ggplot(aes(word, n)) +
    geom_col(fill = fill_color) +
    scale_y_continuous(labels = scales::comma) +
    labs(title = title) +
    ggthemes::theme_fivethirtyeight() +
    coord_flip()
}

gridExtra::grid.arrange(
tidy_trolls_n %>%
  plot_words(word, "troll", "Troll: Top Words"),
tidy_trolls_n %>%
  plot_words(word, "real_tweet", "Real Tweet: Top Words"), ncol = 1)

gridExtra::grid.arrange(
  tidy_trolls_ngrams_n %>%
    filter(!str_detect(ngram, "null|00 00 rt")) %>%
    plot_words(ngram, "troll", "Troll: Top ngrams (3)"),
  tidy_trolls_ngrams_n %>%
    plot_words(ngram, "real_tweet", "Real Tweet: Top ngrams (3)"), ncol = 1)

roc_data <- read_csv("ROC.csv")

roc_data <- roc_data %>%
  select(-1) %>%
  mutate(features = parse_number(model),
    model = factor(model, levels = roc_data %>% distinct(model) %>% pull(), labels = unique(glue::glue("{features}"))))

library(gganimate)
p <- ggplot(roc_data, aes(FPR, TPR, color = model)) + geom_line(size = 1) +
  geom_point(size = 0.9, alpha = 0.4) +
  geom_segment(aes(
    x = 0,
    y = 1,
    xend = 1,
    yend = 1
  ),
  color = "black",
  alpha = 0.2,
  linetype = "dotted") +
  scale_color_viridis_d(name = "Number of Features") +
  labs(x = "1 - Specificity [False Positive Rate]") +
  scale_x_reverse(name = "1 - Specificity [False Positive Rate]",
                  limits = c(1, 0), breaks = seq(0, 1, .02), expand = c(0.001, 0.001)) +
  scale_y_continuous(
    name = "Sensitivity [True Positive Rate]",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = c(0.001, 0.001)
  ) +
  scale_size(range = c(2, 12)) +
  theme_minimal() +
  theme(axis.ticks = element_line(color = "grey80")) +
  transition_states(model, transition_length = 2, state_length = 1)

animate(p, renderer = ffmpeg_renderer())

anim_save("roc.mp4")

time_series_data <- read_csv("twtCount_time.csv") %>%
  select(-1) %>%
  mutate(tweet_time = lubridate::mdy(tweet_time),
         event = case_when(
           tweet_time == "2015-03-18" ~ "Trump's announces candidacy"
         ))

event1 <- "Trump announces candidacy"
event2 <- "Hilary's campaign manager's email divulged on wikileaks"

time_series_data %>%
  filter(tweet_language %in% c("en"),
         lubridate::year(tweet_time) >= 2014) %>%
  ggplot(aes(tweet_time, count, color = tweet_language)) +
  geom_line(color = "#2a3990") +
  scale_y_continuous(labels = scales::comma) +
  labs(x = "Date",
       y = "Count") +
  geom_mark_ellipse(aes(filter = tweet_time == "2015-03-18",
                        description = event1), color = "black") +
  geom_mark_ellipse(aes(filter = tweet_time == "2016-10-06",
                        description = event2), color = "black") +
  ggthemes::theme_fivethirtyeight() +
  theme(legend.position = "none")

pred_2000 <- read_csv("pred_2000.csv") %>%
  select(-1)

pred_5000 <- read_csv("pred_5000.csv") %>%
  select(-1)

probs_2000 <- pred_2000 %>%
  separate(probability, c("real_tweet", "troll"), sep = ",") %>%
  mutate_at(vars(real_tweet, troll), ~ as.double(str_remove(.x, "\\[|\\]")))

probs_5000 <- pred_5000 %>%
  separate(probability, c("real_tweet", "troll"), sep = ",") %>%
  mutate_at(vars(real_tweet, troll), ~ as.double(str_remove(.x, "\\[|\\]")))

roc <- pROC::roc(probs_5000$label, probs_5000$troll)

plot(roc)
