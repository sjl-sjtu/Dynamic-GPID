library("tidyverse")
df <- read_csv("/lustre/home/acct-clsyzs/clsyzs/ukbiobank/test/ukb40687_20002.csv")
findID <- function(i,str){
    return(str %in% df[i,-1])
}
n <- nrow(df)
print(n)

nrow1 <- sapply(seq(1,n),findID,str="1081") #中风
id1 <- df$eid[nrow1]
df1 <- data.frame(FID=id1,IID=id1)
write_tsv(df1,"id_stroke.txt",col_names=FALSE)

id_diff <- setdiff(df$eid,df1$IID)
print(length(id_diff))
id3 <- sample(id_diff,nrow(df1),replace=FALSE)
print(intersect(id1,id3))
df3 <- data.frame(FID=id3,IID=id3)
write_tsv(df3,"id_ctr.txt",col_names=FALSE)

dfa <- bind_rows(df1,df3)
dfa <- dfa%>%arrange(FID)
write_tsv(dfa,"id_slt.txt",col_names=FALSE)

#label
df1$label <- 1
df3$label <- 0
dfid <- bind_rows(df1,df3)
dfid <- dfid%>%mutate_if(is.character, as.double)
write_tsv(dfid,"idLabel.txt")

#phenotypes
df <- read_csv("/lustre/home/acct-clsyzs/clsyzs/ukbiobank/main_data/ukb40687.csv")
feature <- grep("^(4079-|4080-|300[0-9]0-|301[0-7]0-|10000[1-7]-|10001[1-9]-|10002[1-5]-)",colnames(df),value=TRUE)
print(feature)
id <- as.double(dfid$IID)
dfm <- df %>% 
        filter(eid %in% id) %>% 
        rename(FID=eid) %>% 
        mutate(IID=FID) %>% 
        select(FID,IID,all_of(feature)) %>%
        left_join(dfid,by=c("FID","IID"))
write_csv(dfm,"phenoData.csv")

