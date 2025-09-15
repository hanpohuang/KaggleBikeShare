library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(GGally)

## import data
train <- vroom("train.csv")
test <- vroom("test.csv")
sample <- vroom("sampleSubmission.csv")

# peak
glimpse(train)
head(train)
tail(train)



# count vs weather bar plot
weather.plot <-
  ggplot(data = train, aes(x = factor(weather), fill = factor(weather))) +
    geom_bar() +
    scale_fill_manual(
      values = c(
        '1' = 'orange3',
        '2' = 'cyan3',
        '3' = 'maroon4',
        '4' = 'green3'
      )) +
    labs(
      title = "Number of Bikes Rented by Weather Type",
      y = "Bikes Count",
      x = "Weather"
    )

# count vs registered
registered.plot <-
  ggplot(data = train, aes(x = registered, y = count)) + 
    geom_point(pch = 19, color = 'skyblue') +
    geom_smooth(method = "lm" , se = FALSE, color = 'brown') +
    labs(
      title = "Number of Bikes Rented by Number of Registered Users",
      y = 'Bikes Count',
      x = 'Number of Registered User'
    )

# count vs temp scatter plot
temp.plot <-
  ggplot(data = train, aes(x = temp, y = count)) +
    geom_point(pch = 19, color = 'chartreuse3') +
    geom_smooth(method = 'lm', se = FALSE, color = 'darkorange2') +
    labs(
      title = 'Number of Bikes Rented by Temperature (C)',
      y = 'Bikes Count',
      x = 'Temperature (C)'
    )

# count vs windspeed scatter plot
wind.plot <-
  ggplot(data = train, aes(x = windspeed, y = count)) +
    geom_point(pch = 19, color = 'darkmagenta') +
    geom_smooth(method = 'lm', se = FALSE, color = 'darkgoldenrod1') +
    labs(
      title = 'Number of Bikes Rented by Windspeed',
      y = 'Bikes Count',
      x = 'Windespeed'
    )

# display 4 panel plot
four.panel <- (weather.plot + registered.plot)/(temp.plot + wind.plot)

# save the 4 panel plot as an image
ggsave('myplot.png', plot = four.panel, width = 12, height = 8, dpi = 300)

# create pairs plot and save it 
train.pair <- ggpairs(train)
ggsave("pairs.jpg", plot = train.pair, width = 12, height = 8, dpi = 300)



## Data Cleaning
train.t <- train |>
  select(-c(registered, casual)) |>
  mutate(
    log_count = log(count)
  ) |>
  select(-count)

# recipe
my_recipe <- recipe(log_count ~ ., data = train.t) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(season = factor(season)) %>%
  step_mutate(holiday = factor(holiday))

# define a model
my_model <- linear_reg(mode = 'regression', engine = 'lm')

# Combine into a workflow and fit
bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model) %>%
  fit(data = train.t)

# predict using test data
lin_preds <- predict(bike_workflow, new_data = test)

# save it as tibbles
my_preds <- lin_preds %>%
  mutate(pred_count = exp(.pred)) # .pred is the numeric vector, so exp() can be applied


head(my_preds, n = 5)

vroom_write(x=my_preds, file="./My_Preds.csv", delim=",")
