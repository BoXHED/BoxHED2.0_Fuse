packages = c("mboost", "survival", "dplyr", "data.table", "R.utils", "optparse")

for (p in packages){
  if (!p %in% rownames(installed.packages())){
    install.packages(p, repos = "http://cran.us.r-project.org")
  }
}

for (p in packages) {
  library(p, character.only = TRUE)
}

opt_parser = OptionParser(option_list=list(
    make_option(c("--trainDataAddr"), type="character", default="", 
              help="address to training data", metavar="character"),
    make_option(c("--runtimesFile"),    type="character", default="", 
              help="address to runtimes file",  metavar="character")
));
opt = parse_args(opt_parser);


time_mboost <- function(train_data, num_trees, loss){
  last_record = train_data %>% group_by(ID) %>% slice(which.max(t_start))
  first_record = train_data %>% group_by(ID) %>% slice(which.min(t_start))
  first_record$Event_Time = last_record$t_end

  first_record$delta = last_record$delta
  first_record$t_start = NULL
  first_record$t_end = NULL
  
  features = names(first_record)
  features = features[!features%in%c('Event_Time', 'delta', 'ID')]
  fm <- paste0("Surv(Event_Time, delta) ~ ", paste(features, collapse = "+"))
  fm = formula(fm)
  num_event1 = length(which(first_record$Event_Time==1))
  first_record$Event_Time[first_record$Event_Time==1] = first_record$Event_Time[first_record$Event_Time==1] +
    runif(num_event1, -0.02, 0.02)
  dat = first_record
 
  ptm <- proc.time()[[3]]
  
  fit <- mboost::blackboost(fm, data = dat, family = Lognormal(),
                            control=boost_control(mstop = num_trees, trace=T))

  ptm <- proc.time()[[3]]- ptm
  
  return(list(train_time=ptm))
}

nrow_to_sub    = c(
    "2000000"  =  "93193",
    "4000000"  =  "186333",
    "6000000"  =  "279512",
    "8000000"  =  "372658",
    "10000000" =  "465804"
)
runtime_list <- list()
train_data = fread(opt$trainDataAddr)
for(nrow in c("2000000", "4000000", "6000000", "8000000", "10000000")){
  
  rslts = time_mboost(train_data[train_data$ID <= as.numeric(nrow_to_sub[nrow]),], 1, "Lognormal")
  runtime_list <- append(runtime_list, paste(nrow, ":", rslts$train_time, sep=""))
}

writeLines(as.character(runtime_list), file(opt$runtimesFile), sep="\n")