library(tidyverse)

list_of_files <- list.files(pattern="out_chr([1-9]|1[0-9]|2[0-2]).raw",full.names=FALSE)
list_of_files

df <- read_delim(list_of_files[1])
for(i in 2:length(list_of_files)){
  df <- df %>% left_join(read_delim(list_of_files[i]),by=colnames(df)[1:6])
}
# df <- map_dfr(.x=set_names(list_of_files), .f=read_delim, .id="source_file")
# glimpse(df)

dfp <- read_csv("phenoData.csv")
df <- df%>%left_join(dfp,by=c("FID","IID"))

df_pheno <- df %>%
  #  rename_at(vars(contain("4079")),~str_replace(., pattern = "4079", replacement ='DBP')) %>%
  select(IID,matches("(-[0-4].0)$")) %>%
  #rowwise(IID) %>%
  mutate(NArate=rowMeans(is.na(across(-IID)))) %>%
  filter(NArate<0.3) %>%
  missMethods::impute_mean() %>%
  pivot_longer(cols = -c(IID,NArate), 
               names_to = c('.value','times'), 
               names_pattern = '([0-9]*)-([0-4]).0') %>%
  filter(!is.na(times)) %>%
  select(-NArate) %>%
  mutate_at(vars(times),as.numeric) %>%
  arrange(IID,times) %>%
  rename_at(vars(matches("^[0-9]")), ~ paste0("V",.)) %>%
  mutate_at(vars(IID),as.character) %>%
  write_csv("../df_pheno4.csv")

  # simputation::impute_lm(V100018~.-IID-times) %>%
  
  

# df_pheno[,-c(1,2)] <- missMethods::impute_EM(df_pheno[,-c(1,2)])
  

#df_pheno[,-c(1,2)] <- mice::mice(unname(as.matrix(df_pheno[,-c(1,2)])),m=5,method = "rf")$data
  

colMeans(is.na(df_pheno))
hist(df_pheno$NArate)

df_geno <- df %>%
  select(IID,matches("^(rs|[0-9]*:)"),label) %>%
  mutate_at(vars(IID),as.character) %>%
  filter(IID %in% df_pheno$IID) %>%
  arrange(IID) %>%
  missMethods::impute_mode() %>%
  # map_dfc(~ replace_na(.x, rstatix::get_mode(.x)[1])) %>%
  write_csv("../df_geno4.csv")

summary(df_geno$label)


# #######3
# df <- read_csv("../df_pheno_imputed.csv") %>%
#   filter(times==0) %>%
#   left_join(read_csv("../df_geno_imputed.csv")%>%select(IID,label),by="IID")
# 
# 
# 
# df <- read_csv("../df_pheno2.csv") %>%
#   filter(times==0) %>%
#   left_join(read_csv("../df_geno2.csv")%>%select(IID,label),by="IID")%>%
#   select(-IID,-times)
# library(caret)
# index <- createDataPartition(
#   df$label,
#   p = 0.7,
#   list = FALSE
# )
# df$label <- as.factor(df$label)
# train=df[index,]
# test=df[-index,]
# lm = train(label~.,method="glm",family="binomial",data=train)
# pre = predict(lm,train)
# p=table(pre,train$label)
# p[2,2]/(p[1,2]+p[2,2])
# p
# acc=(p[1,1]+p[2,2])/nrow(train)
# acc
# nrow(train[which(train$label==0),])/nrow(train)
# 
# pre = predict(lm,test)
# p=table(pre,test$label)
# p[2,2]/(p[1,2]+p[2,2])
# p
# acc=(p[1,1]+p[2,2])/nrow(train)
# acc
# 
# lm = train(label~.,method="rf",data=train)
