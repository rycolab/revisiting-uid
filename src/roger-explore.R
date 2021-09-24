require(tidyverse)
require(ggplot2)
require(mgcv)
require(lme4)
require(visreg)
require(ggpubr)
dundee_sen <- read_csv("~/tmp/dundee_sentence_uid.txt")
dundee_sen_agg <- dundee_sen %>%
  group_by(log_prob_power_1.0,log_prob_power_1.5,log_prob_power_2.0,word_len_sum,word_len_mean,freq_sum,freq_mean,text_id_,sentence_num_) %>%
  summarize(time_sum=mean(time_sum),time_mean=mean(time_mean))
ggplot(dundee_sen,aes(x=freq_mean,y=..density..)) + geom_density()
ggplot(dundee_sen,aes(x=log_prob_power_1.0,y=..density..)) + geom_density()
ggplot(dundee_sen,aes(x=log_prob_power_2.0,y=..density..)) + geom_density()
ggplot(dundee_sen,aes(x=log_prob_power_3.5,y=..density..)) + geom_density()
ggplot(dundee_sen,aes(x=word_len_mean,y=..density..)) + geom_density()

dundee_fix <- read_csv("~/tmp/dundee_word_uid.txt")
ggplot(dundee_fix,aes(x=log_prob,y=..density..)) + geom_density()

dundee_word_agg <- dundee_fix %>%
  filter(TEXT!=-99) %>%
  group_by(text_id,word,Word_Number,sentence_num,WLEN,WorkerId,log_prob,prev_log_prob,prev2_log_prob,prev3_log_prob,word_len,prev_word_len,freq,prev_freq) %>%
  summarize(time=sum(time)) %>%
  group_by(text_id,word,Word_Number,sentence_num,WLEN,log_prob,prev_log_prob,prev2_log_prob,prev3_log_prob,word_len,prev_word_len,freq,prev_freq) %>%
  summarize(time=mean(time))
ggplot(dundee_word_agg,aes(x=log_prob,y=..density..)) + geom_density()
ggplot(dundee_word_agg,aes(x=freq,y=..density..)) + geom_density()


provo_sen <- read_csv("~/tmp/provo_sentence_uid.txt") %>%
  mutate(sentence_ID=paste("T",text_id_,"S",sentence_num_))
provo_sen_agg <- provo_sen %>%
  group_by(log_prob_power_1.0,log_prob_power_1.5,log_prob_power_2.0,word_len_sum,word_len_mean,freq_sum,freq_mean,text_id_,sentence_num_,sentence_ID) %>%
  summarize(time_sum=mean(time_sum),time_mean=mean(time_mean))

ggplot(provo_sen,aes(x=freq_mean,y=..density..)) + geom_density()
ggplot(provo_sen,aes(x=log_prob_power_1.0,y=..density..)) + geom_density()
ggplot(provo_sen,aes(x=log_prob_power_2.0,y=..density..)) + geom_density()
ggplot(provo_sen,aes(x=log_prob_power_3.5,y=..density..)) + geom_density()
ggplot(provo_sen,aes(x=word_len_mean,y=..density..)) + geom_density()

sen_agg <- provo_sen_agg


ns_word <- read_csv("~/tmp/ns_word_uid.txt")
ns_word_agg <- ns_word %>%
  group_by(text_id,zone,word,sentence_num,ref_token,prev_word,log_prob,prev_log_prob,prev2_log_prob,prev3_log_prob,word_len,prev_word_len,freq,prev_freq) %>%
  summarize(time=mean(time))


provo_word <- read_csv("~/tmp/provo_word_uid.txt")
provo_word_agg <- provo_word %>%
  group_by(text_id,Word_Cleaned,sentence_num,ref_token,prev_word,log_prob,prev_log_prob,prev2_log_prob,prev3_log_prob,word_len,prev_word_len,freq,prev_freq) %>%
  summarize(gaze=mean(IA_FIRST_RUN_DWELL_TIME,na.rm=TRUE))

m.gam1.onlySurprisal <- gam(time_sum ~ s(log_prob_power_1.0,bs="cr",k=20),data=sen_agg)
m.gam1.5.onlySurprisal <- gam(time_sum ~ s(log_prob_power_1.5,bs="cr",k=20),data=sen_agg)
m.gam2.onlySurprisal <- gam(time_sum ~ s(log_prob_power_2.0,bs="cr",k=20),data=sen_agg)
plot(m.gam1.onlySurprisal)
plot(m.gam1.5.onlySurprisal)
plot(m.gam2.onlySurprisal)
summary(m.gam1.onlySurprisal)
summary(m.gam1.5.onlySurprisal)
summary(m.gam2.onlySurprisal)

# notes: additive effect of sentence_num_ doesn't seem to help with model fit

m.gam1 <- gam(time_mean ~ s(log_prob_power_1.0,bs="cr",k=20) + s(freq_mean,bs="cr",k=20) + s(word_len_mean,bs="cr",k=20), data=sen_agg)
m.gam1.5 <- gam(time_mean ~ s(log_prob_power_1.5,bs="cr",k=20) + s(freq_mean,bs="cr",k=20) + s(word_len_mean,bs="cr",k=20), data=sen_agg)
m.gam2 <- gam(time_mean ~ s(log_prob_power_2.0,bs="cr",k=20) + s(freq_mean,bs="cr",k=20) + s(word_len_mean,bs="cr",k=20), data=sen_agg)
plot(m.gam1,ylim=c(-400,400))
plot(m.gam1.5,ylim=c(-400,400))
plot(m.gam2,ylim=c(-400,400))
summary(m.gam1)
summary(m.gam1.5)
summary(m.gam2)


m.null <- lmer(time_mean ~ 1 + (1 | WorkerId_),data=provo_sen,REML=F)
m1 <- lmer(time_mean ~ log_prob_power_1.0 + freq_mean + word_len_mean + time_count_nonzero + (log_prob_power_1.0 | WorkerId_) + (1 | sentence_ID),data=provo_sen,REML=F)
m1.null <- lmer(time_mean ~ 1 + (log_prob_power_1.0 | WorkerId_),data=provo_sen,REML=F)
m1.int <- lmer(time_mean ~ log_prob_power_1.0 + (1 | WorkerId_),data=provo_sen,REML=F)
summary(m1)
anova(m.null,m1)
anova(m1.null,m1)
anova(m.null,m1.int)
m2 <- lmer(time_mean ~ log_prob_power_2.0 + (log_prob_power_2.0 | WorkerId_),data=provo_sen,REML=F)
m2.null <- lmer(time_mean ~ 1 + (log_prob_power_2.0 | WorkerId_),data=provo_sen,REML=F)
m2.int <- lmer(time_mean ~ log_prob_power_2.0 + (1 | WorkerId_),data=provo_sen,REML=F)
anova(m.null,m2)
anova(m2.null,m2)
anova(m.null,m2.int)

m.gam.dundee_word_agg <- gam(time ~ s(log_prob,bs="cr",k=10) + 
                               s(prev_log_prob,bs="cr",k=10) +
                               s(word_len,bs="cr",k=10) +
                               s(prev_word_len,bs="cr",k=10) +
                               s(freq,bs="cr",k=10) +
                               s(prev_freq,bs="cr",k=10),data=dundee_word_agg)
plot(m.gam.dundee_word_agg,ylim=c(-200,200))
plots <- visreg(m.gam.dundee_word_agg,type="contrast",gg=TRUE)
pdf("/tmp/dundee_GAMs.pdf")
for(i in 1:length(plots)) {
  print(plots[[i]] + ggtitle("Dundee"))
}
dev.off()

m.gam.ns_word_agg <- gam(time ~ s(log_prob,bs="cr",k=10) + 
                           s(prev_log_prob,bs="cr",k=10) +
                           s(word_len,bs="cr",k=10) +
                           s(prev_word_len,bs="cr",k=10) +
                           s(freq,bs="cr",k=10) +
                           s(prev_freq,bs="cr",k=10),data=ns_word_agg)
plots <- visreg(m.gam.ns_word_agg,type="contrast",gg=TRUE)
plot(m.gam.ns_word_agg,ylim=c(-200,200))
pdf("/tmp/ns_GAMs.pdf")
for(i in 1:length(plots)) {
  print(plots[[i]] + ggtitle("Natural Stories"))
}
dev.off()

m.gam.provo_word_agg <- gam(gaze ~ s(log_prob,bs="cr",k=10) + 
                           s(prev_log_prob,bs="cr",k=10) +
                           s(word_len,bs="cr",k=10) +
                           s(prev_word_len,bs="cr",k=10) +
                           s(freq,bs="cr",k=10) +
                           s(prev_freq,bs="cr",k=10),data=provo_word_agg)
plots <- visreg(m.gam.provo_word_agg,type="contrast",gg=TRUE)
plot(m.gam.ns_word_agg,ylim=c(-200,200))
pdf("/tmp/provo_GAMs.pdf")
for(i in 1:length(plots)) {
  print(plots[[i]] + ggtitle("Provo"))
}
dev.off()