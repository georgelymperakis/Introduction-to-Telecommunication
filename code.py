#-----------------------------------------------------
#         Εισαγωγή στις Τηλεπικοινωνίες
#    Εργαστηριακή Άσκηση , Ακαδ. Έτος 2020-2021
#
#        ΓΕΩΡΓΙΟΣ ΛΥΜΠΕΡΑΚΗΣ , 03118881
#        ΝΤΑΝΙΕΛΑ ΣΤΟΓΙΑΝ    , 03118140
#
#-----------------------------------------------------
# 
#-----------------------------------------------------
#                     WARNING!
#-----------------------------------------------------
#
#(1):Στο ερώτημα 5 να τοποθετηθεί το path στο οποίο 
#είναι αποθηκευμένο το original sound file.
#
#(2):Στα ερωτήματα όπου υπολογίζεται το BER, ο χρόνος 
#διεκπεραίωσης εκτιμάται από μερικά δευτερόλεπτα έως 
#και μερικά λεπτά (~10 min).
#
#(3):Ο κώδικας υπολογίζει με βάση τα δεδομένα του
#ΑΜ:03118140. Για τον ΑΜ:03118881 τα αντίστοιχα
#δεδομένα υπάρχουν σε σχόλια.
#
#-----------------------------------------------------
#
#-----------------------------------------------------
#ΑΚΟΛΟΥΘΕΙ Ο ΚΩΔΙΚΑΣ ΠΟΥ ΥΛΟΠΟΙΕΙ ΤΑ ΕΡΩΤΗΜΑΤΑ 
#ΣΕ ΓΛΩΣΣΑ Python.
#-----------------------------------------------------


import numpy as np
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt
from numpy import sum,isrealobj,sqrt,abs
from numpy.random import standard_normal
import math
from scipy import special as sp
import sys
from scipy import signal
import binascii
import scipy
from scipy.io import wavfile
import wave
from math import erfc
from scipy.io.wavfile import write
plt.style.use('ggplot')

 
#-------------------------------------------------
#Άσκηση 1 -> Δειγματοληψία
#-------------------------------------------------
 
#Δεδομένα για το σήμα και τις συχνότητες δειγματοληψίες
 
A=1                   
fm=5000.0             
#fm=8000.0
AM = 2+0
#AM = 2+1
fs1 = 20*fm           
fs2 = 100*fm          
Tm=1/fm               
Ts1 = 1/fs1           
Ts2 = 1/fs2           
cycles=4              
 
#---------------------------------------------------
#Δειγματοληψία με συχνότητα 20fm
 
ts1=np.arange(0,cycles*Tm,Ts1)
ts1_p=np.arange(0,cycles*Tm,Ts1/100)
y1=A*np.cos(2*np.pi*fm*ts1)*np.cos(2*np.pi*AM*fm*ts1) 
y1_p=A*np.cos(2*np.pi*fm*ts1_p)*np.cos(2*np.pi*AM*fm*ts1_p)  
plt.plot(ts1_p,y1_p,'--')
plt.stem(ts1,y1,use_line_collection=True)
plt.title("Δειγματοληψία με συχνότητα 20fm")
plt.xlabel("Χρονος (s)")
plt.ylabel("Τιμή Δείγματος (V)")
plt.show()
 
#----------------------------------------------------
#Δειγματοληψία με συχνότητα 100fm
 
ts2=np.arange(0,cycles*Tm,Ts2)
ts2_p=np.arange(0,cycles*Tm,Ts2/100)
y2=A*np.cos(2*np.pi*fm*ts2)*np.cos(2*np.pi*AM*fm*ts2)   
y2_p=A*np.cos(2*np.pi*fm*ts2_p)*np.cos(2*np.pi*AM*fm*ts2_p) 
plt.plot(ts2_p,y2_p,'--')
plt.stem(ts2,y2,use_line_collection=True)
plt.title("Δειγματοληψία με συχνότητα 100fm")
plt.xlabel("Χρονος (s)")
plt.ylabel("Τιμή Δείγματος (V)")
plt.show()
 
#----------------------------------------------------
#Απεικόνιση των δύο δειγματοληπτημένων παραστάσεων από κοινού
 
fs3=1000*fm
 
n3= int(Tm*fs3)
 
x3 = np.linspace(0,4*Tm,num= n3,endpoint=True)
 
y1=A*np.cos(2*np.pi*fm*ts1)*np.cos(2*np.pi*AM*fm*ts1)   
y2=A*np.cos(2*np.pi*fm*ts2)*np.cos(2*np.pi*AM*fm*ts2)   
y3=A*np.cos(2*np.pi*fm*x3)*np.cos(2*np.pi*AM*fm*x3)
 
plt.plot(ts2,y2,'o',ts1,y1,'o',x3,y3,'--')
plt.legend(['Δείγματα με fs = 100fm', 'Δείγματα με fs = 20fm', 'Αρχικό'], loc='best')
plt.title("Δειγματοληψία με συχνότητα 20fm & 100fm")
plt.xlabel("Χρονος (s)")
plt.ylabel("Τιμή Δείγματος (V)")
plt.show()
 
#----------------------------------------------------
#Δειγματοληψία με συχνότητα 5fm
 
fs_b = 5*fm
Ts_b = 1/fs_b
ts_b=np.arange(0,cycles*Tm,Ts_b)
y_b=A*np.cos(2*np.pi*fm*ts_b)*np.cos(2*np.pi*AM*fm*ts_b) 
plt.stem(ts_b,y_b,use_line_collection=True) 
plt.title("Δειγματοληψία με συχνότητα 5fm")
plt.xlabel("Χρονος (s)")
plt.ylabel("Τιμή Δείγματος (V)")
plt.show()

#Άσκηση 2 -> Κβαντοποίηση 
#-------------------------------------------------
 
#Δεδομένα για την υλοποίηση του κβαντιστή
 
b = 5
#b=4                     
U = 1             
N_levels = 2**b
delta = 2*U / N_levels
 
f = 5000 
#f = 8000
Fs = 20*f    
T = 1/f 
 
#--------------------------------------------------
#Υλοποίηση συνάρτηση κβαντιστή που επιστρέφει
#το σήμα κβαντισμένο
 
def quantize(x, S):
    X = x.reshape((-1,1))
    S = S.reshape((1,-1))
    dists = abs(X-S)
    
    nearestIndex = dists.argmin(axis=1)
    quantized = S.flat[nearestIndex]
    
    return quantized.reshape(x.shape)
 
       
t = np.arange(0, T, 1/Fs)
y1=U*np.cos(2*np.pi*f*t)*np.cos(2*np.pi*2*f*t)
 
S = -U + delta/2 + np.arange(N_levels) * delta
 
#Κβαντισμένο σήμα
y1_q = quantize(y1, S)          
n_q = y1 - y1_q               
                
 
l = np.arange(0,T,1/Fs)
plt.xticks(l)
plt.yticks(S,('10000','10001','10011','10010','10110','10111','10101','10100','11100','11101','11111','11110','11010','11011','11001','11000','01000','01001','01011','01010','01110','01111','01101','01100','00100','00101','00111','00110','00010','00011','00001','00000')) 
 
#plt.yticks(S,('1000','1001','1011','1010','1110','1111','1101','1100','0100','0101','0111','0110','0010','0011','0001','0000'))
 
#Απεικόνιση κβαντισμένου σήματος
 
plt.xlabel("Χρόνος (s)")
plt.ylabel("Επίπεδα Κβαντιστή (Gray Coding)")
plt.title("Έξοδος Κβαντιστή")
#plt.grid()
plt.plot(t, y1_q, 'o')
plt.show()
 
 
#-------------------------------------------------
# Ερώτημα β -> i,ii,iii
 
#Συνάρτηση που υπολογίζει απο μία λίστα
#την μέση τιμή της.
def mean_value(x,length):
    res = 0
    for i in range(0,length):
        res += x[i]
    return(res/length)
 
#Συνάρτηση που υπολογίζει απο μια λίστα την
#διασπορά της.
def variance(x,length):
    res = 0
    mean = mean_value(x,length)
    for i in range(0,length):
        res += pow(x[i]-mean,2)
    return(res/(length-1))
        
 
#Μέση ισχύς Σήματος Διακριτού Χρόνου
def spower(x,len):
  res=0
  for i in range(0,len):
    res+=i*i
  return res/len
 
#Τυπική Απόκλιση
s1 = np.sqrt(variance(n_q,10))
s2 = np.sqrt(variance(n_q,20))
 
print(s1)
print(s2)
 
#SNR Κβάντισης
SNR_1 = 10*np.log10(spower(y1_q,10)/variance(n_q,10))
SNR_2 = 10*np.log10(spower(y1_q,20)/variance(n_q,20))
 
print(SNR_1)
print(SNR_2)
 
#-------------------------------------------------
# Ερώτημα γ’
 
A = 5
#A = 8
 
x = np.arange(0, 100)
y = np.array([1,-1,-1,-1,-1, 1,-1,1,1,-1, 1,1,-1,1,1, -1,1,-1,1,1, -1,1,-1,1,-1, -1,1,-1,-1,-1, 1,1,-1,1,-1, 1,1,-1,-1,1, -1,1,1,1,-1, -1,-1,-1,1,1, -1,-1,-1,-1,-1, -1,-1,-1,1,1, -1,1,-1,1,-1, 1,1,-1,-1,1, 1,1,-1,1,1, -1,1,-1,-1,-1, -1,1,1,1,-1, -1,1,-1,1,1, 1,1,-1,1,-1, 1,-1,1,1,-1])
plt.xlim(0, 100)
plt.ylim(-A-1, A+1)
 
#Bit stream με κωδικοποίηση γραμμής Polar NRZ
 
plt.step(x, A*y)

 
plt.xlabel("Χρόνος (s)")
plt.ylabel("Πλάτος Παλμών (V)")
plt.title("Ροή Μετάδοσης")
plt.show()
 
#-------------------------------------------------
#Άσκηση 3 -> BPAM-AWGN
#-------------------------------------------------
 
#Υποερώτημα α'
 
#Δεδομένα 
 
T_b = 0.5
l = [0 , 1]
lnew = [1,-1]
seq = []
seq_new = []
A = 5
#A = 8
#-------------------------------------------------
#Παραγωγή τυχαίας ακολουθίας bits
 
for i in range(1,47):
    seq.append(random.choice(l))
 
for i in range(1,47):
    seq_new.append(random.choice(lnew))
 
x = []
for i in range(0,46):
    x.append(i*T_b)
 
y = A*np.array(seq)
y_new = A*np.array(seq_new)
 
#Απεικόνιση ακολουθίας bits
 
plt.step(x, y_new)

 
plt.title("Τυχαία ακολουθία από Bits κατά B-PAM")
plt.xlabel("Χρόνος (s)")
plt.ylabel("Πλάτος Παλμών (V)")
plt.show()
 
 
x_c = []
for i in range(0,4600):
    x_c.append(i*(T_b/100))
 
t = []
for i in range(0,len(y_new)):
    for j in range(0,100):
        t.append(y_new[i])
 
y_c = np.array(t)
 
#-------------------------------------------------
#Υποερώτημα β'
 
#Κατασκευή του διαγράμματος αστερισμού
 
x_cons = [-(A**2)*T_b ,+(A**2)*T_b]
y_cons = [0 ,0]
 
plt.vlines(0,-5,5,'k','--')
plt.scatter(x_cons, y_cons, s=400 , marker='*')
plt.text(-(A**2)*T_b -2,-5,"Energy = 12.5 J")
plt.text((A**2)*T_b -2,-5,"Energy = 12.5 J")
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Αστερισμός BPAM")
plt.show()
 
#-------------------------------------------------
#Υποερώτημα γ'
 
#Συνάρτηση που προσθέτει awgn θόρυβο
 
def produce_noise(length, snr):
    noise_mean_value = 0
    noise_stand_dev=1
    noise = np.random.normal(noise_mean_value, noise_stand_dev ,length) 
    return noise
 
 
#Διαγράμματα με θόρυβο 5 και 15 db
 
y_5 = y_c + pow(10,-5/10)*produce_noise(len(y_c),5)
y_15 = y_c + pow(10,-15/10)*produce_noise(len(y_c),15)
 
fig, axs = plt.subplots(3)
axs[0].step(x,y_new)
axs[0].set_title("Μετάδοση Χωρίς Θόρυβο")
axs[0].set_xlabel("Χρόνος (s)")
axs[0].set_ylabel("Πλάτος Παλμών (V)")
axs[1].plot(x_c,y_5)
axs[1].set_title("Μετάδοση Με Θόρυβο 5 dB")
axs[1].set_xlabel("Χρόνος (s)")
axs[1].set_ylabel("Πλάτος Παλμών (V)")
 
 
axs[2].plot(x_c,y_15,'g')
axs[2].set_title("Μετάδοση Με Θόρυβο 15 dB")
axs[2].set_xlabel("Χρόνος (s)")
axs[2].set_ylabel("Πλάτος Παλμών (V)")
 
fig.tight_layout() 
 
plt.show()
#-------------------------------------------------
#Υποερώτημα δ'
 
#Διαγράμματα αστερισμών για μιγαδικό θόρυβο 5 και 15 db
 
y1_c = y + pow(10,-5/10)*(produce_noise(len(y),5) + 1j*produce_noise(len(y),5))
 
y2_c = y + pow(10,-15/10)*(produce_noise(len(y),15) + 1j*produce_noise(len(y),15))
 
k=[0,A]
l=[0,0]
 
#Διάγραμμα για 15db

plt.plot(y2_c.real,y2_c.imag, '*')
plt.plot(k,l, '*')
plt.xticks([-1,0,1,2,3,4,5,6],[-3.5/2.5,-2.5/2.5,-1.5/2.5,-0.5/2.5,0.5/2.5,1.5/2.5,2.5/2.5,3.5/2.5])
plt.ylim(-3, 3)
plt.vlines(2.5,-3,3,'g',':')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Αστερισμός BPAM με AWGN 15 dB")
plt.show()

#Διάγραμμα για 5db

plt.plot(y1_c.real,y1_c.imag, '*')
plt.plot(k,l, '*')
plt.xticks([-3,-2,-1,0,1,2,3,4,5,6,7,8],[-5.5/2.5,-4.5/2.5,-3.5/2.5,-2.5/2.5,-1.5/2.5,
-0.5/2.5,0.5/2.5,1.5/2.5,2.5/2.5,3.5/2.5,4.5/2.5,5.5/2.5])
plt.ylim(-3, 3)
plt.vlines(2.5,-3,3,'g',':')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Αστερισμός BPAM με AWGN 5 dB")          
plt.show()

 
 
#-------------------------------------------------
#Υποερώτημα ε'
 
#Δεδομένα
 
 
l = [0 , 1]
signal = []
p = pow(10,6)
 
#Παραγωγή τυχαίας ακολουθίας bits
 
for i in range(0,p):
    signal.append(random.choice(l))
            
 
def cal_N(snr,E):
    N = E/(pow(10,snr/10))
    return N
 
 
def cal_items(s,s_n,b):
    count = 0
    for i in range(0,p) : 
        if s[i] == 0:
            if s_n[i]>b:
                count+=1
        if s[i] == 1:
            if s_n[i]<=b:
                count+=1
    return count
 
BER = []
error = []
dis = 0.5
 
#Υπολογισμός πειραματικού Bit Error Rate
      
for snr_db in range(0, 16):
    signal_noise = signal + pow(10,-snr_db/10)*produce_noise(len(signal),snr_db)
    BER.append(cal_items(signal,signal_noise,dis))
    
 
x_snr = []
 
 
for i in range(0,16):    
    x_snr.append(i)
 
 
for i in range(0,16):
    BER[i]=BER[i]/p
    
#Υπολογισμός θεωρητικού Bit Error Rate
 
theoryBER = np.zeros(len(x_snr),float)
for i in range(0,len(x_snr)):
    theoryBER[i] = 0.5*erfc(np.sqrt(2*10**(x_snr[i]/10)))

plt.plot(x_snr,BER,'ro',)
plt.plot(x_snr,theoryBER,'k')
plt.xticks(x_snr)
plt.title('BER of BPAM With AWGN')
plt.xlabel('Eb/No(dB)')
plt.ylabel('BER')
plt.yscale('log')
plt.xscale('linear')
plt.legend(["Πειραματικό","Θεωρητικό"],loc ="best")
plt.show()
 
#-------------------------------------------------
#Ερώτημα 4 -> QPSK 
#-------------------------------------------------
 
#Δεδομένα
 
freq=5
#freq = 8
T_b = 0.5
l = [0 , 180]
seq = []
A = 5
#A =8
Fs = 276.0
Ts = 1.0/Fs
 
#Παραγωγή τυχαίας ακολουθίας bits
 
for i in range(1,47):
    seq.append(random.choice(l))
 
t = np.arange(0,20,Ts)
 
#Κατασκευή του σήματος QPSK
 
bit_arr = np.array(seq)
samples_per_bit = 20*Fs/bit_arr.size 
dd = np.repeat(bit_arr, samples_per_bit)
y= A*np.sin(2 * np.pi * (freq) * t+(np.pi*dd/180))
 
plt.plot(t,y)
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("QPSK Διαμόρφωση")
plt.show()
 
 
#-------------------------------------------------
#Υποερώτημα α’ ->Constellation
 
A = 5
#A = 8
T_b = 0.5
 
 
#Κατασκευή διαγραμμάτων αστερισμών
 
x_cons = [math.sqrt((A**2)*T_b) ,+math.sqrt((A**2)*T_b),-math.sqrt((A**2)*T_b) ,-math.sqrt((A**2)*T_b)]
y_cons = [math.sqrt((A**2)*T_b)  ,-math.sqrt((A**2)*T_b)  ,-math.sqrt((A**2)*T_b)  ,math.sqrt((A**2)*T_b) ]
plt.vlines(0,-5,5,'k','--')
plt.hlines(0,-5,5,'k','--')
 
plt.scatter(x_cons, y_cons, s=400 , marker='*')
 
plt.title('QPSK Constellation Diagram')
plt.xlabel('In Phase Component φ1')
plt.ylabel('Quadrate Component φ2')
 
string1 = "00"
string2 = "01"
string3 = "10"
string4 = "11"
 
plt.annotate(string4,((A**2)*T_b,(A**2)*T_b),(10,10))
plt.annotate(string2,(-(A**2)*T_b,(A**2)*T_b),(-10,10))
plt.annotate(string3,((A**2)*T_b,-(A**2)*T_b),(10,-10))
plt.annotate(string1,(-(A**2)*T_b,-(A**2)*T_b),(-10,-10))
plt.show()
 
 
#-------------------------------------------------
#Υποερώτημα β’
 
 
 
#Δεδομένα
 
T_b = 0.5
A = 1
l = [A + 1j*A , A - 1j*A, - A + 1j*A, - A - 1j*A]
seq = []
snr = 15
 
#Τυχαία ακολουθία bits 
 
for i in range(1,100):
    seq.append(random.choice(l))
 
#Συνάρτηση παραγωγής θορύβου 
def produce_noise(length, snr):
    noise_mean_value = 0
    noise_stand_dev=1
    noise = np.random.normal(noise_mean_value, noise_stand_dev ,length) 
    return noise
 
#Σήμα στο οποίο έχει προστεθεί θόρυβος 5 και 15 db
 
sig_5 = seq + pow(10,-5/10)*(produce_noise(len(seq),5) + 1j*produce_noise(len(seq),5))
 
sig_15 = seq + pow(10,-15/10)*(produce_noise(len(seq),15) + 1j*produce_noise(len(seq),15)) 
 
y=np.array(seq)
 
#Διαγράμματα αστερισμών για 5 και 15 db θόρυβο
 
plt.plot(sig_5.real, sig_5.imag, '*')
plt.plot(y.real, y.imag, '*')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Διάγραμμα Αστερισμού QPSK με AWGN 5 dB")
plt.vlines(0,-2,2,'k','--')
plt.hlines(0,-2,2,'k','--')
plt.show()
 
plt.plot(sig_15.real, sig_15.imag, '*')
plt.plot(y.real, y.imag, '*')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Διάγραμμα Αστερισμού QPSK με AWGN 15 dB")
plt.vlines(0,-2,2,'k','--')
plt.hlines(0,-2,2,'k','--')
plt.show()
 
 
 
#-------------------------------------------------
#Υποερώτημα γ’
 
l = [-1 , 1]
signal = []
p = pow(10,6)
 
#Τυχαία ακολουθία bits
 
for i in range(0,p):
    signal.append(random.choice(l)+ 1j*random.choice(l))
 
#Συνάρτηση θορύβου
 
def produce_noise(length, snr):
    noise_mean_value = 0
    noise_stand_dev=1
    noise = np.random.normal(noise_mean_value, noise_stand_dev ,length) 
    return noise
 
def calculate_items(s,s_n):
    count = 0
    for i in range(0,len(s)):
        if np.real(s[i]) == -1 and np.real(s[i])==-1:
            if np.real(s_n[i])< 0 and np.imag(s_n[i])<0:
                count+=1
                
        if np.real(s[i]) == -1 and np.imag(s[i])==1:
            if np.real(s_n[i])<0 and np.imag(s_n[i])>0:
                count+=1
                
        if np.real(s[i]) == 1 and np.imag(s[i])==1:
            if np.real(s_n[i])>0 and np.imag(s_n[i])>0:
                count+=1
                
        if np.real(s[i]) == 1 and np.imag(s[i])==-1:
            if np.real(s_n[i])>0 and np.imag(s_n[i])<0:
                count+=1
    
    return (len(s) - count)
 
#Υπολογισμός πειραματικού bit error rate
 
BER = []
 
for snr_db in range(0, 16):
    signal_noise = signal + pow(10,-snr_db/10)*(produce_noise(len(signal),snr_db) + 1j*produce_noise(len(signal),snr_db))
    BER.append(calculate_items(signal,signal_noise))
 
x_snr = []
 
 
for i in range(0,16):    
    x_snr.append(i)
 
 
for i in range(0,16):
    BER[i]=BER[i]/p
 
#Υπολογισμός θεωρητικού bit error rate    
 
theoryBER = np.zeros(len(x_snr),float)
 
for i in range(0,len(x_snr)):
    theoryBER[i] = 0.5*erfc(np.sqrt(10**(x_snr[i]/10)))
 
 
plt.plot(x_snr,BER,'ro')
plt.plot(x_snr,theoryBER,'y')
plt.xticks(x_snr)
plt.title('BER of QPSK With AWGN')
plt.xlabel('Es/No(dB)')
plt.ylabel('BER')
plt.yscale('log')
plt.legend(["Πειραματικό","Θεωρητικό"],loc ="best")
plt.xscale('linear')
plt.show()

 
 
#-------------------------------------------------
#Υποερώτημα δ’
 
#-------------------------------------------------
#Υποερώτημα δ’ -> (i)
 
#Μετατροπή κειμένου σε binary
 
byte_array = ("We can think of a discrete source as generating the message, symbol by symbol. It will choose successive symbols according to certain probabilities depending, in general, on preceding choices as well as the particular symbols in question. A physical system, or a mathematical model of a system which produces such a sequence of symbols governed by a set of probabilities, is known as a stochastic process. We may consider a discrete source, therefore, to be represented by a stochastic process. Conversely, any stochastic process which produces a discrete sequence of symbols chosen from a finite set may be considered a discrete source.").encode()
#byte_array = ("We now consider the case where the signal is perturbed by noise during transmission or at one or the other of the terminals. This means that the received signal is not necessarily the same as that sent out by the transmitter. Two cases may be distinguished. If a particular transmitted signal always produces the same received signal, i.e., the received signal is a definite function of the transmitted signal, then the effect may be called distortion. If this function has an inverse β no two transmitted signals producing the same received signal β distortion may be corrected, at least in principle, by merely performing the inverse functional operation on the received signal.").encode()
 
 
binary_int = int.from_bytes(byte_array, "big")
binary_string = bin(binary_int)
 
#-------------------------------------------------
#Υποερώτημα δ’ -> (ii)
 
bit_seq = list(binary_string)
bit_seq.remove('b')
 
string = "" 
quant = []
p=len(bit_seq)
#Κβάντιση 8 bit του σήματος
 
for i in range (0,p,8):
  for j in range(0,8,1):
    string += bit_seq[i+j]      
  quant.append(int(string ,2))
  string = ""
 
#Απεικόνιση του κβαντισμένου σήματος 
pa = int(p/8) 
z = np.linspace(0,pa-1,pa)
plt.plot(z,quant,'.')
plt.grid()
plt.xlabel("Δείγμα")
plt.ylabel("Επίπεδα Κβάντισης")
plt.title("Έξοδος Κβαντιστή")
plt.show()
 
 
pre_signal = []
for bit in bit_seq:
  pre_signal.append(int(bit,2))
 
signal = []
for i in range(0,len(pre_signal),2):
    if pre_signal[i]==0:
      pre_signal[i]=-1
    if pre_signal[i+1]==0:
      pre_signal[i+1]=-1
    signal.append(complex(pre_signal[i],pre_signal[i+1]))
 
 
 
#-------------------------------------------------
#Υποερώτημα δ’ -> (iii)
 
#Δεδομένα 
T = 0.5
f=2
A = 1
l=[0,1]
snr = 15
k=0
 
qpsk_const = []
 
#Συνάρτηση υπολογισμού της φάσης για την QPSK διαμόρφωση 
 
def theta1(x,y):
  x=int(x)
  y=int(y)
  if x==1 and y ==1:
    return np.pi/4
  elif x==1 and y ==0:
    return -np.pi/4
  elif x==0 and y ==1:
    return 3*np.pi/4
  elif x==0 and y ==0:
    return -3*np.pi/4 
 
def color(x,y):
  x=int(x)
  y=int(y)
  if x==1 and y ==1:
    return 'm'
  elif x==1 and y ==0:
    return 'b'
  elif x==0 and y ==1:
    return 'r'
  elif x==0 and y ==0:
    return 'c' 
 
theta = float 
 
#Κατασκευή QPSK σήματος
 
for i in range(0,20,2):
  x = np.linspace(i*T,(i+2)*T,50)
  theta = theta1(bit_seq[i],bit_seq[i+1])
  y=A*math.sqrt(2)*np.cos(2*np.pi*f*x - float(theta))
  plt.plot(x,y,color(bit_seq[i],bit_seq[i+1]))  
plt.ylabel("Τάση (V)")
plt.title("Διαμορφωμένο Σήμα Χωρίς Θόρυβο")
plt.xlabel("Χρόνος (s)")  
plt.show()
 
 
 
#-------------------------------------------------
#Υποερώτημα δ’ -> (iv)
 
#Αναπαράσταση σημάτων με θόρυβο 5 και 15 db αντίστοιχα 
 
for i in range(0,20,2):
    x = np.linspace(i*T,(i+2)*T,50)
    theta = theta1(bit_seq[i],bit_seq[i+1])
    y=A*math.sqrt(2)*np.cos(2*np.pi*f*x - float(theta))
    k = pow(10,-5/10)*np.random.normal(0,1,len(y)) + y 
    plt.plot(x,k,color(bit_seq[i],bit_seq[i+1]))
 
plt.xlabel("Χρόνος (s)")
plt.ylabel("Τάση (V)")
plt.title("Διαμορφωμένο Σήμα με θόρυβο 5 dB")
plt.show()
 
for i in range(0,20,2):
    x = np.linspace(i*T,(i+2)*T,50)
    theta = theta1(bit_seq[i],bit_seq[i+1])
    y=A*math.sqrt(2)*np.cos(2*np.pi*f*x - float(theta))
    k = pow(10,-15/10)*np.random.normal(0,1,len(y)) + y 
    plt.plot(x,k,color(bit_seq[i],bit_seq[i+1]))
    
 
plt.xlabel("Χρόνος (s)")
plt.ylabel("Τάση (V)")
plt.title("Διαμορφωμένο Σήμα με θόρυβο 15 dB")
plt.show()
 
 
 
#----------------------------------------------------------------------
#Υποερώτημα δ’ -> (v),(vi)
 
def theta2(x,y):
    x=int(x)
    y=int(y)
    if x==1 and y ==1:
        qpsk = A + 1j*A
        return qpsk
    if x==1 and y ==0:
        qpsk = A - 1j*A
        return qpsk
    if x==0 and y ==1:
        qpsk = -A + 1j*A
        return qpsk
    if x==0 and y ==0:
        qpsk = -A - 1j*A
        return qpsk
 
 
for i in range(0,5096,2):
    qpsk_const.append(theta2(bit_seq[i],bit_seq[i+1]))
 
#Κατασκευή διαγραμμάτων αστερισμού για 5 και 15 db
 
snr_db = 5
qpask_c1 = qpsk_const + pow(10,-snr_db/10)*(produce_noise(len(qpsk_const),snr_db) + 1j*produce_noise(len(qpsk_const),snr_db))
 
snr_db = 15
qpask_c2 = qpsk_const + pow(10,-snr_db/10)*(produce_noise(len(qpsk_const),snr_db) + 1j*produce_noise(len(qpsk_const),snr_db))
 
#Αναπαράσταση των διαγραμμάτων αστερισμού
  
qpsk_c = np.array(qpsk_const)
plt.show()
plt.plot(qpsk_c.real,qpsk_c.imag,'*')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Διάγραμμα Αστερισμού QPSK")
plt.vlines(0,-2,2,'k','--')
plt.hlines(0,-2,2,'k','--')
 
qpask_c1 = np.array(qpask_c1)
plt.show()
plt.plot(qpask_c1.real,qpask_c1.imag,'.',qpsk_c.real,qpsk_c.imag,'.')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Διάγραμμα Αστερισμού QPSK με AWGN 5 dB")
plt.vlines(0,-2,2,'k','--')
plt.hlines(0,-2,2,'k','--')
 
qpask_c2 = np.array(qpask_c2)
plt.show()
plt.plot(qpask_c2.real,qpask_c2.imag,'.',qpsk_c.real,qpsk_c.imag,'.')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Διάγραμμα Αστερισμού QPSK με AWGN 15 dB")
plt.vlines(0,-2,2,'k','--')
plt.hlines(0,-2,2,'k','--')
plt.show() 
 
#-------------------------------------------------
#Υποερώτημα δ’ -> (vi)
 
#Συνάρτηση αποδιαμόρφωσης
 
def demodulation(s):
  demod = []
  for c in s:
    if np.real(c) > 0 and np.imag(c) > 0:
      demod.append(1+1j)
    if np.real(c) < 0 and np.imag(c) > 0:
      demod.append(-1+1j)
    if np.real(c) > 0 and np.imag(c) < 0:
      demod.append(1-1j)
    if np.real(c) < 0 and np.imag(c) < 0:
      demod.append(-1-1j)
    
  return demod
 
de_qpsk1 = demodulation(qpask_c1)
de_qpsk2 = demodulation(qpask_c2)
 
#Συνάρτηση εύρεσης σφάλματος μεταξύ δύο σημάτων
 
def find_error(signal,singal_noise):
  count = 0 
  for i in range(0,len(signal)):
    if signal[i] != singal_noise[i]:
      count+=1
  return count
 
#Υπολογισμός πειραματικού bit error rate
 
BER = []
BER.append(find_error(signal,de_qpsk1))
BER.append(find_error(signal,de_qpsk2))
 
p = len(signal)
 
for i in range(len(BER)):
  BER[i]/=p
 
x_snr = [5,15]
 
#Υπολογισμός θεωρητικού bit error rate
 
theoryBER =[]
for i in x_snr:
    theoryBER.append(0.5*erfc(np.sqrt(10**(i/10)))) 
plt.plot(x_snr,BER,'ro')
plt.plot(x_snr,theoryBER,'go')
 
plt.title('BER of QPSK With AWGN')
plt.xlabel('Es/No(dB)')
plt.ylabel('BER')
plt.yscale('log')
plt.xscale('linear')
plt.text( 9, pow(10,-23),"Πειραματικό_BER(SNR=15dB) = 0")
plt.legend(["Πειραματικό","Θεωρητικό"],loc ="best")
plt.show()
 
print(theoryBER)
print(BER)
#-------------------------------------------------
#Υποερώτημα δ’ -> (vii)
 
#Αποδιαμόρφωση σήματος με θόρυβο
 
def calc_dist(x):
  x1 = math.sqrt((x.real - 1)**2 + (x.imag - 1)**2) 
  x2 = math.sqrt((x.real - (-1))**2 + (x.imag - 1)**2)
  x3 = math.sqrt((x.real - (-1))**2 + (x.imag - (-1))**2)
  x4 = math.sqrt((x.real - 1)**2 + (x.imag - (-1))**2)
  x_min = min(x1,x2,x3,x4)
  if (x_min==x1):
    return "11"
  elif (x_min == x2):
    return "01"  
  elif (x_min == x3):
    return "00" 
  elif (x_min == x4):
    return "10" 
 
#Αποδιαμόρφωση σήματος με 5 db θόρυβο 
 
demod = []
flat_demod=[] 
for i in range(0,2548):
  demod.append(calc_dist(qpask_c1[i]))
 
demod_char = [list(i) for i in demod]
for sublists in demod_char:
    for t in sublists:
        flat_demod.append(t)
 
#Μετατροπή binary ακολουθίας σε κείμενο
 
bit_list = []
bit_list_temp = ""
for i in range (0,len(flat_demod),8):
  for j in range (i,i+8):
    bit_list_temp += flat_demod[j]
  bit_list.append(bit_list_temp)
  bit_list_temp = ""
 
binary_to_string1 = ""
for i in range (0,637):
  binary_to_string1 += chr(int(bit_list[i], 2)) 
print(binary_to_string1)
 
#Αποδιαμόρφωση σήματος με 15 db θόρυβο
 
demod1 = []
flat_demod1=[]
for i in range(0,2548):
  demod1.append(calc_dist(qpask_c2[i]))
 
demod1_char = [list(i) for i in demod1]
for sublists in demod1_char:
    for t in sublists:
        flat_demod1.append(t)
 
#Μετατροπή binary ακολουθίας σε κείμενο 
 
bit_list = []
bit_list_temp = ""
for i in range (0,len(flat_demod1),8):
  for j in range (i,i+8):
    bit_list_temp += flat_demod1[j]
  bit_list.append(bit_list_temp)
  bit_list_temp = ""
 
binary_to_string2 = ""
for i in range (0,637):
  binary_to_string2 += chr(int(bit_list[i], 2)) 
print(binary_to_string2)

#-------------------------------------------------
#Ερώτημα 5 
#-------------------------------------------------
#Υποερώτημα α’

#Αναπαράσταση κυματομορφή του ήχου

#Για να λειτουργήσει ο κώδικας παρακαλώ τοποθετήστε το path του αρχείου #στον υπολογιστή σας

song = r"C:\Users\daniela\Downloads\soundfile1_lab2.wav"
fs, data = wavfile.read(song)
x=np.linspace(0,len(data),329413)
plt.title("Κυματομορφή Αρχικού Σήματος")
plt.xlabel("Χρόνος")
plt.ylabel("Πλάτος")

plt.plot(x,data)
plt.show()

#-------------------------------------------------
#Υποερώτημα β’

 
#Συνάρτηση κβαντιστή
 
def quantize(x, S):
    X = x.reshape((-1,1))
    S = S.reshape((1,-1))
    dists = abs(X-S)
    
    nearestIndex = dists.argmin(axis=1)
    quantized = S.flat[nearestIndex]
    
    return quantized.reshape(x.shape)

#Δεδομένα 
 
U=max(data)
b = 8   
N_levels = 2**b
delta = 2*U / N_levels
S = -U + delta/2 + np.arange(N_levels) * delta

#Διάγραμμα κβαντισμένου σήματος

quant = quantize(data,S)
plt.title("Έξοδος 8Bits Κβαντιστή")
plt.xlabel("Χρόνος")
plt.ylabel("Επίπεδα Κβάντισης") 
plt.plot(x,quant,'b')
plt.show()

#--------------------------------------------------------
#Υποερώτημα γ’

#Συνάρτηση μετατροπής binary αριθμών σε κωδικοποίηση gray
 
def binary_to_gray(n):
    n = int(n, 2) 
    n ^= (n >> 1)
    x = str(bin(n)[2:]) 
    x = x.zfill(8) 
    return x
  
stream_gray = []
int_p = [] 
for q in quant:
    i = int(q/delta-0.5)
    stream_gray.append(binary_to_gray(bin(i+128)))  
stream = []
 
for q in stream_gray:
    for i in range(0,8):
        stream.append(int(q[i]))
 
stream_com = []  
stream1 = []
for q in stream:
  stream1.append(q)
 
for i in range(0,len(stream1),2):
    if stream1[i]==0:
      stream1[i]=-1
    if stream1[i+1]==0:
      stream1[i+1]=-1
    stream_com.append(complex(stream1[i],stream1[i+1]))
 
signal = []
for i in stream_com:
  signal.append(i)
  

 
#Δεδομένα

T = 0.5
f=2
A = 1
l=[0,1]
snr = 15
k=0

#Συνάρτηση υπολογισμού γωνίας για QPSK σήμα
 
qpsk_const = []

def theta1(x,y):
    if x==1 and y ==1:
        return np.pi/4
    elif x==1 and y ==0:
        return -np.pi/4
    elif x==0 and y ==1:
        return 3*np.pi/4
    elif x==0 and y ==0:
        return -3*np.pi/4
    else: 
        return 0 
 
def color(x,y):
  if x==1 and y ==1:
    return 'm'
  elif x==1 and y ==0:
    return 'b'
  elif x==0 and y ==1:
    return 'r'
  elif x==0 and y ==0:
    return 'c' 
 
theta = float 
 
#Αναπαράσταση QPSK σήματος

for i in range(0,20,2):
  x = np.linspace(i*T,(i+2)*T,50)
  theta = theta1(stream[i],stream[i+1])
  y=A*math.sqrt(2)*np.cos(2*np.pi*f*x - float(theta))
  plt.plot(x,y,color(stream[i],stream[i+1]))  
plt.title("Διαμορφωμένο Σήμα Χωρίς Θόρυβο")
plt.xlabel("Χρόνος (s)")
plt.ylabel("Πλάτος Σήματος (V)")     
plt.show()
 
 
#-----------------------------------------------------------------
#Υποερώτημα δ’

#Συνάρτηση παραγωγής θορύβου
 
def produce_noise(length, snr):
    noise_mean_value = 0
    noise_stand_dev=1
    noise = np.random.normal(noise_mean_value, noise_stand_dev ,length) 
    return noise

#Υπολογισμός και αναπαράσταση σημάτων QPSK με θόρυβο
 
for i in range(0,20,2):
    x = np.linspace(i*T,(i+2)*T,50)
    theta = theta1(stream[i],stream[i+1])
    y=A*math.sqrt(2)*np.cos(2*np.pi*f*x - float(theta))
    k = pow(10,-4/10)*np.random.normal(0,1,len(y)) + y 
    plt.plot(x,k,color(stream[i],stream[i+1]))
plt.title("Διαμορφωμένο Σήμα με 5 dB Θόρυβο")
plt.xlabel("Χρόνος (s)")
plt.ylabel("Πλάτος Σήματος (V)")
plt.show()
 
for i in range(0,20,2):
    x = np.linspace(i*T,(i+2)*T,50)
    theta = theta1(stream[i],stream[i+1])
    y=A*math.sqrt(2)*np.cos(2*np.pi*f*x - float(theta))
    k = pow(10,-14/10)*np.random.normal(0,1,len(y)) + y 
    plt.plot(x,k,color(stream[i],stream[i+1]))
plt.title("Διαμορφωμένο Σήμα με 15 dB Θόρυβο")
plt.xlabel("Χρόνος (s)")
plt.ylabel("Πλάτος Σήματος (V)") 
plt.show()
 
 
 
#----------------------------------------------------------
#Υποερώτημα ε’

#Συνάρτηση τοποθέτησης bits στο διάγραμμα αστερισμών
""" 
def theta2(x,y):
    x=int(x)
    y=int(y)
    if x==1 and y ==1:
        qpsk = A + 1j*A
        return qpsk
    if x==1 and y ==0:
        qpsk = A - 1j*A
        return qpsk
    if x==0 and y ==1:
        qpsk = -A + 1j*A
        return qpsk
    if x==0 and y ==0:
        qpsk = -A - 1j*A
        return qpsk
"""
#Κατασκευή και αναπαράσταση διαγραμμάτων αστερισμών για θόρυβο 4,14 db 
 
for i in range(0,2635304,2):
    qpsk_const.append(theta2(stream[i],stream[i+1]))
 
snr_db = 4
qpask_c1 = qpsk_const + pow(10,-snr_db/10)*(produce_noise(len(qpsk_const),snr_db) + 1j*produce_noise(len(qpsk_const),snr_db))
 
snr_db = 14
qpask_c2 = qpsk_const + pow(10,-snr_db/10)*(produce_noise(len(qpsk_const),snr_db) + 1j*produce_noise(len(qpsk_const),snr_db))
   
qpsk_c = np.array(qpsk_const)
plt.plot(qpsk_c.real,qpsk_c.imag,'.')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Αστερισμός QPSK")
plt.show()

#Διάγραμμα για θόρυβο 4 db
 
qpask_c1 = np.array(qpask_c1)
plt.plot(qpask_c1.real,qpask_c1.imag,'.',qpsk_c.real,qpsk_c.imag,'.')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Αστερισμός QPSK με AWGN 4 db")
plt.show()

#Διάγραμμα για θόρυβο 14 db
 
qpask_c2 = np.array(qpask_c2)
plt.plot(qpask_c2.real,qpask_c2.imag,'.',qpsk_c.real,qpsk_c.imag,'.')
plt.xlabel("Συμφασική Συνιστώσα")
plt.ylabel("Ορθογώνια Συνιστώσα")
plt.title("Αστερισμός QPSK με AWGN 14 db")
plt.show() 
 

#-------------------------------------------------------------------
#Υποερώτημα στ’

#Συνάρτηση αποδιαμόρφωσης

def demodulation(s):
  demod = []
  for c in s:
    if np.real(c) > 0 and np.imag(c) > 0:
      demod.append(1+1j)
    if np.real(c) < 0 and np.imag(c) > 0:
      demod.append(-1+1j)
    if np.real(c) > 0 and np.imag(c) < 0:
      demod.append(1-1j)
    if np.real(c) < 0 and np.imag(c) < 0:
      demod.append(-1-1j)
    
  return demod

de_qpsk1 = demodulation(qpask_c1)
de_qpsk2 = demodulation(qpask_c2)
 
#Υπολογισμός σφάλματος

def find_error(signal,singal_noise):
  count = 0 
  for i in range(0,len(signal)):
    if signal[i] != singal_noise[i]:
      count+=1
  return count 

#Υπολογισμός πειραματικού bit error rate 
 
BER = []
BER.append(find_error(signal,de_qpsk1))
BER.append(find_error(signal,de_qpsk2))
 
p = len(signal)
 
for i in range(len(BER)):
  BER[i]/=p
 



#Υπολογισμός θεωρητικού bit error rate 
 
theoryBER =[]
print(BER)
x_snr = [4,14]
for i in x_snr:
    theoryBER.append(0.5*erfc(np.sqrt(10**(i/10))))
 
plt.plot([4,14],BER,'ro')
plt.plot(x_snr,theoryBER,'go')
print(theoryBER)
plt.title('BER of QPSK With AWGN')
plt.xlabel('Es/No(dB)')
plt.ylabel('BER')
plt.yscale('log')
plt.xscale('linear')
plt.legend(["Πειραματικό","Θεωρητικό"],loc ="best")
plt.show()


#------------------------------------------------------------
#Υποερώτημα ζ

de_qpsk1 = demodulation(qpask_c1)
de_qpsk2 = demodulation(qpask_c2)
 
de_bits2= []
 
#Ανακατασκευή σήματος με θόρυβο (14db)

for d in de_qpsk2:
  if np.real(d)==1:
    de_bits2.append(1)
  if np.real(d)==-1:
    de_bits2.append(0)
  if np.imag(d)==1:
    de_bits2.append(1)
  if np.imag(d)==-1:
    de_bits2.append(0)
 
 
de_strings2=[]
s = ""
 
for i in range(0,len(de_bits2),8):
  s=""
  for j in range(0,8):
    s+=str(de_bits2[i+j])
  de_strings2.append(s)

#Μετατροπή bits με gray code σε κανονικά bits 
 
def gray_to_binary(n):
    n = int(n, 2) # convert to int
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return bin(n)[2:]
 
de_signal2=[]
 
for i in de_strings2:
  de_signal2.append(gray_to_binary(i))
 
 
 
de_level2=[]
for i in de_signal2:
  de_level2.append(int(i,2)+128)
 
 
de_quant2=[]
 
for i in de_level2:
  de_quant2.append(0.5*delta*(2*i+1))
#---------------------------------------------------

#Ανακατασκευή σήματος χωρίς θόρυβο (4db)

de_bits1= []
 
for d in de_qpsk1:
  if np.real(d)==1:
    de_bits1.append(1)
  if np.real(d)==-1:
    de_bits1.append(0)
  if np.imag(d)==1:
    de_bits1.append(1)
  if np.imag(d)==-1:
    de_bits1.append(0)
 
 
 
de_strings1=[]
s = ""
 
for i in range(0,len(de_bits1),8):
  s=""
  for j in range(0,8):
    s+=str(de_bits1[i+j])
  de_strings1.append(s)

#Μετατροπή bits με gray code σε κανονικά bits 
 
de_signal1=[]
 
for i in de_strings1:
  de_signal1.append(gray_to_binary(i))
 
 
 
de_level1=[]
for i in de_signal1:
  de_level1.append(int(i,2)+128)
 
 
de_quant1=[]
 
for i in de_level1:
  de_quant1.append(0.5*delta*(2*i+1))


#Μετατροπή bit array σε αρχείο ήχου

samplerate = 44100
de_quant2 = np.array(de_quant2)
signal2 = de_quant2 * (2**8 - 1)/ np.max(np.abs(de_quant2))
audio2 = signal2.astype(np.int8)

 
de_quant1 = np.array(de_quant1)
signal1 = de_quant1 * (2**8 - 1)/ np.max(np.abs(de_quant1))
audio1 = signal1.astype(np.int8)
write("song_4db.wav", samplerate, audio1)
write("song_14db.wav", samplerate, audio2)

#Απεικόνιση τελικών κυματομορφών αρχείων 
 
x=np.linspace(0,len(de_quant1))
plt.plot(x,de_quant1)
plt.title("Κυματομορφή Αποδιαμορφωμένου Σήματος με Θόρυβο 4 dB")
plt.xlabel("Χρόνος")
plt.ylabel("Πλάτος")
plt.show()
 
x=np.linspace(0,len(de_quant2))
plt.plot(x,de_quant2)
plt.title("Κυματομορφή Αποδιαμορφωμένου Σήματος με Θόρυβο 14 dB")
plt.xlabel("Χρόνος")
plt.ylabel("Πλάτος")
plt.show()
