#!/usr/bin/env python
# coding: utf-8

# In[ ]:


• # recursive
def fibonacci(n):
    if(n<=1):
        return n
    else:
        return(fibonacci(n-1)+fibonacci(n-2))
n=int(input())
print("fibonacci sequence")
for i in range(n):
    print(fibonacci(i))


• # non-recursive
nterms=int(input("enter the number of terms"))
n1,n2=0,1
count=0
if nterms<=0:
    print("enter the positive integer")
elif nterms==1:
    print("fibonacci sequence upto",nterms,":")
    print(n1)
else:
    print("fibonacci sequence")
while count<nterms:
    print(n1)
    nth=n1+n2
    n1=n2
    n2=nth
    count+=1 

