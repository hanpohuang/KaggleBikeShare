library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(GGally)
library(glmnet)

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




#### Workflows
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

my_Preds <- exp(lin_preds)

## Format the Predictions for Submission to Kaggle
kaggle_submission2 <- my_Preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables4
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)5
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)6
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle7


head(kaggle_submission2, n = 5)

vroom_write(x=kaggle_submission2, file="./My_Preds.csv", delim=",")



#### Penalized Regression

## import data
train <- vroom("train.csv")
test <- vroom("test.csv")

## Data Cleaning
train.t <- train |>
  select(-c(registered, casual)) |>
  mutate(
    log_count = log(count)
  ) |>
  select(-count)

## recipe
my_recipe <- recipe(log_count ~ ., data = train.t) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>% # This step removes the original datetime column
  step_mutate(season = factor(season)) %>%
  step_mutate(holiday = factor(holiday)) %>%
  step_mutate(workingday = factor(workingday)) %>%
  
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) #make mean 0, sd = 1 so everything is on the same scale

## Penalized regression model
# model 1
preg_model1 <- linear_reg(penalty = 1, mixture = 0.5) %>% # pen = 1, mix = 0.5
  set_engine('glmnet') # function to fit in R

preg_wf1 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model1) %>%
  fit(data = train.t)

mypreds1 <- exp(predict(preg_wf1, new_data = test))

# Format the Predictions for Submission to Kaggle
kaggle.plr.1 <- mypreds1 %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables4
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)5
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)6
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle7

vroom_write(x=kaggle.plr.1, file="./kaggle_plr_1.csv", delim=",")


# model 2
preg_model2 <- linear_reg(penalty = 1, mixture = 1) %>% #pen = 1, mix = 1 (Lasso)
  set_engine('glmnet') # function to fit in R

preg_wf2 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model2) %>%
  fit(data = train.t)

mypreds2 <- exp(predict(preg_wf2, new_data = test))

# Format the Predictions for Submission to Kaggle
kaggle.plr.2 <- mypreds2 %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables4
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)5
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)6
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle7

vroom_write(x=kaggle.plr.2, file="./kaggle_plr_2.csv", delim=",")


# model 3
preg_model3 <- linear_reg(penalty = 1, mixture = 0) %>% #pen = 1, mix = 0 (Ridge)
  set_engine('glmnet') # function to fit in R

preg_wf3 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model3) %>%
  fit(data = train.t)

mypreds3 <- exp(predict(preg_wf3, new_data = test))

# Format the Predictions for Submission to Kaggle
kaggle.plr.3 <- mypreds3 %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables4
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)5
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)6
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle7

vroom_write(x=kaggle.plr.3, file="./kaggle_plr_3.csv", delim=",")


# model 4
preg_model4 <- linear_reg(penalty = 0.5, mixture = 0.5) %>% #pen = 0.5, mix = 0.5 
  set_engine('glmnet') # function to fit in R

preg_wf4 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model4) %>%
  fit(data = train.t)

mypreds4 <- exp(predict(preg_wf4, new_data = test))

# Format the Predictions for Submission to Kaggle
kaggle.plr.4 <- mypreds4 %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables4
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)5
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)6
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle7

vroom_write(x=kaggle.plr.4, file="./kaggle_plr_4.csv", delim=",")


# model 5
preg_model5 <- linear_reg(penalty = 2, mixture = 0.5) %>% #pen = 2, mix = 0.5 
  set_engine('glmnet') # function to fit in R

preg_wf5 <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model5) %>%
  fit(data = train.t)

mypreds5 <- exp(predict(preg_wf5, new_data = test))

# Format the Predictions for Submission to Kaggle
kaggle.plr.5 <- mypreds5 %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables4
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)5
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)6
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle7

vroom_write(x=kaggle.plr.5, file="./kaggle_plr_5.csv", delim=",")
