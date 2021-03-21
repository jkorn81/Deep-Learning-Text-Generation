#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [-] - Set Working Directory ----
setwd("C:/Users/jonat/Desktop/project.deeptxtgen")#create a data folder in wd...
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
library(reticulate)
use_python("C:/Users/jonat/Anaconda3/python.exe")
library(keras)
#library(tidyverse) 
library(dplyr)
library(readr)
library(stringr)
library(purrr)
library(tokenizers)
library(tm)
#library(tfdatasets)
library(data.table)
library(harrypotter) 
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# Parameters --------------------------------------------------------------
#Set parameters:
FLAGS <- flags(
  flag_integer('filters_cnn', 32),
  flag_integer('filters_lstm', 128),
  flag_numeric('reg1', 5e-4),
  flag_numeric('reg2', 0.01),
  flag_numeric('batch_size',20),
  flag_numeric('maxlen', 10),
  flag_numeric('steps', 5),
  flag_numeric('embedding_dim', 1000),
  flag_numeric('kernel', 5),
  flag_numeric('leaky_relu', 0.50),
  flag_numeric('epochs', 100),
  flag_numeric('lr', 0.002)
)
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# - [3] - Harry Potter Text ----
list_of_txt <- list.files(path = "data/.", recursive = TRUE,
                          pattern = "\\.txt$", 
                          full.names = TRUE)

for(i in seq_along(length(list_of_txt))) {
  text = lapply(list_of_txt, readtext::readtext)
}
for(i in seq_along(length(text))) {
  string = rbindlist(text)
}
str(string)

# - [4] - Pre-process Text ----
data.text = string$text[1]
data.text = tolower(data.text)
data.text = iconv(data.text, "latin1", "ASCII", sub = " ")
#data.text = tm::removeWords(data.text, stopwords("SMART"))
data.text = gsub("^NA| NA ", " ", data.text)
data.text = tm::removeNumbers(data.text)
#data.text = tm::removePunctuation(data.text)
data.text = tm::stripWhitespace(data.text)
str(data.text)
data.text[1]

# - [5] - Prepare Tokenized Forms of Each Text ----
text =  tokenize_regex(data.text, simplify = TRUE)
print(sprintf("corpus length: %d", length(text)))

vocab <- gsub("\\s", "", unlist(text)) %>%
  unique() %>%
  sort()
print(sprintf("total words: %d", length(vocab))) 

sentence <- list()
next_word <- list()
list_words <- data.frame(word = unlist(text), stringsAsFactors = F)
j <- 1

for (i in seq(1, length(list_words$word) - FLAGS$maxlen - 1, by = FLAGS$steps)){
  sentence[[j]] <- as.character(list_words$word[i:(i+FLAGS$maxlen-1)])
  next_word[[j]] <- as.character(list_words$word[i+FLAGS$maxlen])
  j <- j + 1
}
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
model = load_model_hdf5("model_final.h5", compile = TRUE)
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
sample_mod <- function(preds, temperature = 1){
  preds <- log(preds)/temperature
  exp_preds <- exp(preds)
  preds <- exp_preds/sum(exp(preds))
  
  rmultinom(1, 1, preds) %>% 
    as.integer() %>%
    which.max()
}

for(diversity in c(0.2, 0.5, 1, 1.2)){
  
  cat(sprintf("diversity: %f ---------------\n\n", diversity))
  
  start_index <- sample(1:(length(text) - FLAGS$maxlen), size = 1)
  sentence <- text[start_index:(start_index + FLAGS$maxlen - 1)]
  generated <- ""
  
  for(i in 1:200){
    
    x <- sapply(vocab, function(x){
      as.integer(x == sentence)
    })
    x <- array_reshape(x, c(1, dim(x)))
    
    preds <- predict(model, x)
    next_index <- sample_mod(preds, diversity)
    nextword <- vocab[next_index]
    
    generated <- str_c(generated, nextword, sep = " ")
    sentence <- c(sentence[-1], nextword)
    
  }
  
  cat(generated)
  cat("\n\n")
  
}