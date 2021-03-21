#######################################################################################
# - [-] - Set Working Directory ----
setwd("C:/Users/jonat/Desktop/project.deeptxtgen")#create a data folder in wd...
library(tfruns)
#Model_Reports...
training_run("01_paratrain.r")
training_run("02_ensemble.r")
training_run("03_test.ensemble.r")
#latest_run()
#view_run("runs/2020-04-15T02-00-09Z")
