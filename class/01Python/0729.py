# -*- coding: utf-8 -*-

#미니실습 1
def min_of(s):
    minimum=s[0]
    
    for i in range(1,len(s)):
        if s[i]<minimum:
            minimum=s[i]
            
    return minimum

#미니실습 2
def reverse1():
    num=int(input('원소 수 : '))
    x=[None]*num
    
    for i in range(num):
        x[i]=int(input(f'x[{i}]값 입력 : '))
        
    print('Reversing...')

    for i in range(num//2):
        x[i],x[num-i-1]=x[num-i-1],x[i] 
    
    for i in range(num):
        print(f'x[{i}]={x[i]}')   
        
    return
    
#미니실습 3
def if_cnt_seq_search(a,key):
    i=0
    cnt=0
    
    while True:
        cnt+=1
        if i==len(a):
            i=-1
            break
        
        cnt+=1
        if a[i]==key:
            break
        
        i+=1
    
    return (i, cnt)

def if_cnt_seq_search_sentinel(b,key):
    a=b.copy()
    a.append(key)
    
    i=0
    cnt=0

    while True:
        cnt+=1
        if a[i]==key:
            break

        i+=1
        
    cnt+=1
    if i==len(b):
        i=-1
        
    return (i, cnt)
   

#Binary Search
def bin_seracy(a,key):
    # a는 오름차순 정렬되어 있어야 함 
    left=0
    right=len(a)-1
    
    while True:
        mid=(left+right)//2
        
        if a[mid]==key:
            return mid
        elif a[mid]<key:
            left=mid+1
        elif a[mid]>key:
            right=mid-1
            
        if left>right:
            break
    
    return -1

#실습 1
def isPrime(num):
    cnt=0
    
    for i in range(1,num+1):
        if num%i==0:
            cnt+=1
            
        if cnt>2:
            break
        
    if(cnt==2):
        return True
    else:
        return False    
    
#실습 2
def reverse2():
    x=[]
    num=0
    
    while True :
        data=input(f'x[{num}]값 입력 (STOP press X) : ')
        
        if(data=='X'):
            break
        elif(data.isdigit()):
            x.append(int(data))
            num+=1
        else:
            print('다시 입력하시오.')
    
    for i in range(num//2):
        x[i],x[num-1-i]=x[num-1-i],x[i]
            
    print('Reversing...')
    
    for i in range(num):
        print(f'x[{i}]={x[i]}')   
    
    return

#실습 3
def find_idx():
    arr=[]
    num=0
    idx=-1
    while True :
        data=input(f'arr[{num}]값 입력 (STOP press X) : ')
        
        if(data=='X'):
            break
        elif(data.isdigit()):
            arr.append(int(data))
            num+=1
        else:
            print('다시 입력하시오.')
    
    target=int(input('검색할 값: '))
    
    for i in range(num):
        if arr[i]==target :
            idx=i
            break
    
    print(f'검색할 값 {target}은 {idx}에 위치합니다')

    return i 

#실습 4
def find_max():
    arr=[]
    num=0
    
    while True :
        data=input(f'arr[{num}]값 입력 (STOP press X) : ')
        
        if(data=='X'):
            break
        elif(data.isdigit()):
            arr.append(int(data))
            num+=1
        else:
            print('다시 입력하시오.')
    
    Max=arr[0]
    idx=0
    
    for i in range(1,num):
        if arr[i]>Max:
            Max=arr[i]
            idx=i
    
    print(f'최대값은 {Max}이고, {idx}번째에 위치합니다')
    
    return (Max, idx)

#추가실습 0
def Q0():
    N, X = map(int, input().split())  # N, X 를 입력받음
    data = list(map(int, input().split())) # 리스트를 입력받음
        
    answer = []
        
    for i in range(N):
        if data[i]<X:
            answer.append(data[i])
    
    for i in range(N):
        print(i, end=' ')
        
    return

#추가실습 1
def Q1(lottos, win_nums):
    answer = []
    
    zeros=0
    Min=0
    Max=0
    
    for n in lottos:
        if n==0:
            zeros+=1
        elif n in win_nums:
            Min+=1 
    
    Max=Min+zeros
    
    if Max<2:
        Max=1
    if Min<2:
        Min=1
    
    answer=[7-Max, 7-Min]
    
    return answer

#추가실습 2
def Q2(numbers):
    answer = 45
    sum=0
    
    for n in numbers:
        sum+=n
    
    answer-=sum
    
    return answer

#추가실습 3
def Q3(store, customer):
    answer = []
    '''
    for item in customer:
        if item in store:
            answer.append('yes')
        else:
            answer.append('no')
    '''
    #store는 정렬되어 있기에 BinarySearch로
    for item in customer:
        left=0
        right=len(store)-1
        
        while True:
            mid=(left+right)//2
            
            if store[mid]==item:
                answer.append('yes')
                break
            elif store[mid]<item:
                left=mid+1
            elif store[mid]>item:
                right=mid-1
            
            if left>right:
                answer.append('no')
                break
                 
    return answer

#추가실습 4
def Q4(arr):
    answer=0
    length=len(arr)
    arr.sort()
    Max=arr[length-1]
    
    for num in range(Max, Max**length+1, Max):
        cnt=0

        for i in arr:
            if(num%i!=0):
                break
            else:
                cnt+=1
        
        if(cnt==length):
            answer=num
            break
    
    return answer

#추가실습 5
def Q5(n, s):
    answer = []
    
    if(s<2):
        answer=[-1]
    else:
        answer=[s//2, s-s//2]
        
    return answer

#추가실습 6
def Q6(arr):
    length=len(arr)
    
    if(length<2):
        arr=[-1]
    else:
        Min=arr[0]
        idx=0
        
        #Min=min(arr)
        for i in range(1,length):
            if(arr[i]<Min):
                Min=arr[i]
                idx=i
        
        #arr.remove(Min)
        arr.pop(idx)
        
    return arr


#추가실습 7
def Q7(arr):
    answer = []

    length=len(arr)
    prev=arr[0]
    
    for i in range(1,length):
        if(arr[i]!=prev):
            answer.append(prev)
            prev=arr[i]
        else:
            continue
    answer.append(prev)
    
    return answer

##################################################
 
def main():
    # 문제별 함수에 맞게 파라미터 선언 및 입력 필요함
    arr = [1,1,3,3,0,1,1]
    answer = solution(arr)
    print(answer)
    ##############################################
    
main()

##################################################