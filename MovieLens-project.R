#### MovieLens project ####
# Author: Kiko Núñez 
# Start Date: 09/08/2019
# End Date: 13/10/2019

## Load libraries needed:
library(stringr)
library(dplyr)
library(caret)
library(ggplot2)
library(grid)
library(scales)

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#### 1. Initial analysis ####
# Visual inspection:
str(edx)

#### 2. Preprocessing ####
## Extract year of rating from timestamp:
edx <- transform(edx, timestamp = as.POSIXct(timestamp, origin = "1970-01-01"))
edx$yearRating <- as.integer(format(edx$timestamp, '%Y')) # Add yearRating column to edx dataset

## Add year of movie column:
edx$yearMovie <- as.integer(sub("\\).*", "", sub(".*\\(", "", edx$title)))

## Genre bagging:
GenresBags <- unique(unlist(str_split(edx$genres, "\\|"))) # Split genre value by '|'
print(GenresBags)
print(paste("Movies in the dataset have", 
            length(GenresBags), "different types of genres."))

GenresEdx <- matrix(, nrow(edx), length(GenresBags)) # set-up a matrix for bagging genres
colnames(GenresEdx) <- GenresBags

## Populate the matrix of genres (this may take a while):
pb <- txtProgressBar(min = 0, max = length(GenresBags), style = 3)
for (i in 1:length(GenresBags)) {
  GenresEdx[grep(GenresBags[i],edx$genres), i] <- 1
  setTxtProgressBar(pb, i)
}
GenresEdx[is.na(GenresEdx)] <- 0

edx <- cbind(edx, GenresEdx)

## Drop useless columns: timestamp and title:
edx_clean <- edx %>% select(-c(timestamp, title))

#### 3. Understanding the dataset ####
## Overall ratings distribution:
edx_clean %>% ggplot(aes(x=factor(rating))) + 
  geom_bar(color="steelblue", fill="steelblue") + 
  labs(x = "Ratings", y = "Counts") +
  scale_y_continuous(labels = comma)

summary(edx_clean$rating)

## Ratings (counts and mean value) by users:
edx_clean %>% group_by(userId) %>% summarise(Users = n()) %>%
  ggplot(aes(Users)) + geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  scale_x_log10() + labs(title = "Number of ratings per user") + 
  xlab("Number of ratings") + ylab("Number of users")

edx_clean %>% group_by(userId) %>% summarise(meanRating = mean(rating)) %>%
  ggplot(aes(meanRating)) + geom_histogram(bins = 50, fill = "salmon4", color = "white") +
  labs(title = "Mean ratings per users") + 
  xlab("Mean Rating") + ylab("Number of users")
  
## Ratings (counts and mean value) by movies:
edx_clean %>% group_by(movieId) %>% summarise(Movies = n()) %>%
  ggplot(aes(Movies)) + geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  scale_x_log10() + labs(title = "Number of ratings per movie") +
  xlab("Number of ratings") + ylab("Number of movies")

edx_clean %>% group_by(movieId) %>% summarise(meanRatingMovie = mean(rating)) %>%
  ggplot(aes(meanRatingMovie)) + geom_histogram(bins = 50, fill = "salmon4", color = "white") +
  labs(title = "Mean ratings per movie") +
  xlab("Mean rating") + ylab("Number of movies")

## Ratings (counts and mean value) by movie antiquity:
edx_clean %>% mutate(yearDiff = yearRating - yearMovie) %>% 
  group_by(yearDiff) %>% summarise(Difference = n()) %>%
  ggplot(aes(yearDiff, Difference)) + geom_bar(stat = "identity", fill = "steelblue", color = "white") +
  scale_y_continuous(labels = comma) + labs(title = "Number of ratings per movie antiquity") +
  xlab("Movie antiquity") + ylab("Number of ratings") 

edx_clean %>% mutate(yearDiff = yearRating - yearMovie) %>%
  group_by(yearDiff) %>% summarise(meanRatingYearDiff = mean(rating)) %>% 
  ggplot(aes(yearDiff, meanRatingYearDiff)) + geom_point(color = "salmon4") + 
  geom_smooth(method = "loess", formula = y ~ x) +
  labs(title = "Mean rating per movie antiquity", 
       x = "Movie antiquity (in years)", y = "Mean rating")

## Ratings (counts and mean values) by genres (aggregated):
edx_clean %>% group_by(genres) %>% summarise(Genres = n()) %>%
  ggplot(aes(Genres)) + geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  scale_x_log10(labels = comma) + labs(title = "Number of ratings per genre group") +
  xlab("Number of ratings") + ylab("Number of genre groups")

edx_clean %>% group_by(genres) %>% summarise(meanRatingGenres = mean(rating)) %>%
  ggplot(aes(meanRatingGenres)) + geom_histogram(bins = 50, fill = "salmon4", color = "white") +
  labs(title = "Mean ratings per genres (aggregated)") +
  xlab("Mean rating") + ylab("Number of genres")

## Ratings (counts and mean values) by genres (atomic):
p_countGenres <- NULL
p_ratingGenres <- NULL

for (i in 1:length(GenresBags)) { # Calculate counts and mean ratings per genre
  index <- which(edx_clean[GenresBags[i]] == 1)
  p_ratingGenres <- append(p_ratingGenres, mean(edx_clean[index, "rating"]))
  p_countGenres <- append(p_countGenres, length(index))
}
names(p_ratingGenres) <- GenresBags
names(p_countGenres) <- GenresBags

# Plot the results
par(mai = c(1.8, 1, 1, 1))
barplot(p_countGenres/1000000, ylim = c(-5, 5), axes = FALSE, border = NA,
        col = "steelblue", las = 2, main = "Number of ratings and mean ratings by genre")
barplot(-p_ratingGenres, add = TRUE, axes = FALSE, col = "salmon4", border = NA, names.arg = NA)
axis(2, at = seq(-5, 5, 1), 
     labels = c(rev(seq(0, 5, 1)), seq(1, 5, 1)), las = 2)
mtext("Mean", 2, line = 3, at = -2.5, col = "salmon4")
mtext("Number (Mill.)", 2, line = 3, at = 2.5, col = "steelblue")

## Ratings (count and mean values) by year of movie:
edx_clean %>% group_by(yearMovie) %>% summarise(Years = n()) %>%
  ggplot(aes(yearMovie, Years)) + geom_bar(stat = "identity", fill = "steelblue", color = "white") +
  labs(title = "Number of ratings per year of movie") +
  xlab("Year") + ylab("Number of ratings") + scale_y_continuous(labels = comma)

edx_clean %>% group_by(yearMovie) %>% summarise(meanRatingYear = mean(rating)) %>%
  ggplot(aes(yearMovie, meanRatingYear)) + geom_point(color = "salmon4") + 
  geom_smooth(method = "loess", formula = y ~ x) + labs(title = "Mean ratings per year of movie") +
  xlab("Year") + ylab("Mean rating")

#### 4. Building the model to predict ratings ####
## Baseline (mean) model:
# Baseline calculation:
Baseline <- mean(edx_clean$rating)
print(paste("Baseline model (average): ", Baseline))

## Model includeing user effect:
# Penalty term due to user effect (p_user):
meanUsers <- edx_clean %>% group_by(userId) %>% 
  summarise(p_user = mean(rating - Baseline))
meanUsers %>% ggplot(aes(p_user)) + 
  geom_histogram(bins = 50, fill = "darkgreen", color = "white") +
  labs(title = "Effect of users") +
  xlab("Penalty due to user effect") + ylab("Frequency") 

## Model including user and movie effect:
# Penalty term due to movie effect: 
meanMovies <- edx_clean %>% left_join(meanUsers, by = "userId") %>%
  group_by(movieId) %>% summarise(p_movie = mean(rating - Baseline - p_user))
meanMovies %>% ggplot(aes(p_movie)) + 
  geom_histogram(bins = 50, fill = "darkgreen", color = "white") +
  labs(title = "Effect of movies") +
  xlab("Penalty due to movie effect") + ylab("Frequency") 

## Model including user, movie and antiquity:
# Penalty term due to antiquity effect: 
meanDiffYear <- edx_clean %>% mutate(diffYear = yearRating - yearMovie) %>%
  left_join(meanUsers, by = "userId") %>% left_join(meanMovies, by = "movieId") %>%
  group_by(diffYear) %>% summarise(p_diffYear = mean(rating - Baseline - p_user - p_movie))
meanDiffYear %>% ggplot(aes(p_diffYear)) + 
  geom_histogram(bins = 50, fill = "darkgreen", color = "white") +
  labs(title = "Effect of movie antiquity") +
  xlab("Penalty due to year of movie antiquity effect") + ylab("Frequency") 

## Model including user, movie, antiquity and genre:
# Penalty term due to genre effect: 
meanGenre <- edx_clean %>% mutate(diffYear = yearRating - yearMovie) %>%
  left_join(meanUsers, by = "userId") %>% left_join(meanMovies, by = "movieId") %>% 
  left_join(meanDiffYear, by = "diffYear") %>% group_by(genres) %>% 
  summarise(p_genres = mean(rating - Baseline - p_user - p_movie - p_diffYear))
meanGenre %>% ggplot(aes(p_genres)) + 
  geom_histogram(bins = 50, fill = "darkgreen", color = "white") +
  labs(title = "Effect of movie genre") +
  xlab("Penalty due to movie genre") + ylab("Frequency") 

#### 5. Apply trained model to validation dataset ####
### First perform the same preprocessing:
## Extract year of rating from timestamp:
validation <- transform(validation, timestamp = as.POSIXct(timestamp, origin = "1970-01-01"))
validation$yearRating <- as.integer(format(validation$timestamp, '%Y')) # Add yearRating column to validation dataset

## Add year of movie column:
validation$yearMovie <- as.integer(sub("\\).*", "", sub(".*\\(", "", validation$title)))

## Genre bagging:
GenresVal <- matrix(, nrow(validation), length(GenresBags)) # set-up a matrix for bagging genres
colnames(GenresVal) <- GenresBags

pb <- txtProgressBar(min = 0, max = length(GenresBags), style = 3)
for (i in 1:length(GenresBags)) {
  GenresVal[grep(GenresBags[i],validation$genres), i] <- 1
  setTxtProgressBar(pb, i, title = "Populating genres")
}
GenresVal[is.na(GenresVal)] <- 0

validation <- cbind(validation, GenresVal)

## Drop useless columns (timestamp and title):
validation_clean <- validation %>% select(-c(timestamp, title))

## Now calculate RMSE of the model:
# RMSE of baseline model:
RMSE_Baseline <- RMSE(validation_clean$rating, Baseline)
print(paste("RMSE in baseline model: ", RMSE_Baseline))

# RMSE including user effect:
y_hat_users <- validation_clean %>% left_join(meanUsers, by = "userId") %>%
  mutate(pred_rating = Baseline + p_user)
RMSE_Users <- RMSE(validation_clean$rating, y_hat_users$pred_rating)
print(paste("RMSE adding user effect: ", RMSE_Users))

# RMSE including user and movie effect:
y_hat_movies <- validation_clean %>% left_join(meanUsers, by = "userId") %>%
  left_join(meanMovies, by = "movieId") %>% 
  mutate(pred_rating = Baseline + p_user + p_movie)
RMSE_UsersMovies <- RMSE(validation_clean$rating, y_hat_movies$pred_rating)
print(paste("RMSE adding movie and user effects: ", RMSE_UsersMovies))

# RMSE including user, movie and antiquity effects:
y_hat_diffYear <- validation_clean %>% mutate(diffYear = yearRating - yearMovie) %>% 
  left_join(meanUsers, by = "userId") %>% left_join(meanMovies, by = "movieId") %>% 
  left_join(meanDiffYear, by = "diffYear") %>%
  mutate(pred_rating = Baseline + p_user + p_movie + p_diffYear)
RMSE_UsersMoviesDiffYear <- RMSE(validation_clean$rating, y_hat_diffYear$pred_rating)
print(paste("RMSE adding user, movie and antiquity effects: ", RMSE_UsersMoviesDiffYear))

# RMSE including user, movie, antiquity and genre effects:
y_hat_genre <- validation_clean %>% mutate(diffYear = yearRating - yearMovie) %>% 
  left_join(meanUsers, by = "userId") %>% left_join(meanMovies, by = "movieId") %>% 
  left_join(meanDiffYear, by = "diffYear") %>% left_join(meanGenre, by = "genres") %>%
  mutate(pred_rating = Baseline + p_user + p_movie + p_diffYear + p_genres)
RMSE_UsersMoviesDiffYearGenre <- RMSE(validation_clean$rating, y_hat_genre$pred_rating)
print(paste("RMSE adding user, movie, antiquity and genre effects: ", RMSE_UsersMoviesDiffYearGenre))
