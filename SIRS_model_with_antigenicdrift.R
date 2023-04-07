# time series
step = 7
#in weeks
year = 365
num_year = 15
t = seq(0, year*num_year , by = step)

# parameters
#average infection rate
beta0 = 0.09
#amplitude of sinosuoidal wave
beta1_set = 0.25
#controls sinosuoidal infection phase
omega = (2*pi)/365
#recovery rate
gamma = 0.075
#rate of return to suscepability
lambda_set = 0.015

#add in random antigenic drift 
countdown = 0



# equations 

dS = function(S,I,R,time,lambda, beta1){(-beta_t(time, beta1)*S*I)/N + lambda*R}
dI = function(S,I,time,beta1){(beta_t(time,beta1)*S*I)/N - gamma*I}
dR = function(I,R,lambda){gamma*I - lambda*R}
beta_t = function(time,beta1){beta0*(1 + beta1*sin(omega*time))}

#solve ODEs using Euler method
#generate empty lists 
S = c()
I = c()
R = c()
#set intitial values
S[1] = 999
I[1] = 1
R[1] = 0
N = S[1] + I[1] + R[1]
set.seed(12)


for (i in 1:(length(t)-1)){
  if (countdown < 52) {
    countdown = countdown+1
  }
  if (countdown == 52){
    beta1 = runif(1,0.15,0.75)
    lambda = runif(1,0.010,0.030)
  } 

  S[i+1] = S[i] + step*dS(S[i],I[i],R[i],t[i],lambda, beta1)
  I[i+1] = I[i] + step*dI(S[i],I[i],t[i], beta1)
  R[i+1] = R[i] + step*dR(I[i],R[i], lambda)
  
  if (countdown == 52){
    countdown = 0
  }
}

output = data.frame(S = S, I = I, R = R, T = t)

ggplot(data = output) +
  geom_line(aes(T,I), color = 'red') +
  #geom_line(aes(T,S), color = 'blue') +
  #geom_line(aes(T,R), color = 'green') +
  theme(legend.position = 'right')

##write.csv(output, "C:\\Users\\Alfred\\OneDrive\\Documents\\Important Documents\\Data_Validation_SIRS\\output.csv")
