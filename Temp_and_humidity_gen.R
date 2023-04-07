# time series
step = 7
#in weeks
year = 365
num_year = 15
n = 52 * num_year
t = seq(0, year*num_year , by = step)
a <- 25
b <- 1/58
c <- 10
d <- 1/24
c.unif <- runif(n)
c.norm <- rnorm(n)
amp <- 5
amp2 <- 5
offset <- 13

#temp data
set.seed(1)
y1 <- a*sin(b*t-4)+c.norm*amp 


#precipitation 
set.seed(2)
y2 <- c*sin(d*t-4)+ offset +c.norm*amp2

#offset temp data

set.seed(1)
y3 <- a*sin(b*t-13)+c.norm*amp 


output$Temp <- y1
output$Temp_off <- y3
output$Precip <- y2
output$percent_ili <- output$I /10


ggplot(data = output) +
  geom_line(aes(T,percent_ili), color = 'black') +
  #geom_line(aes(T,Temp), color = 'red') +
  geom_line(aes(T,Temp_off), color = 'orange') +
  #geom_line(aes(T, Precip), color = 'blue') +
  theme(legend.position = 'right')