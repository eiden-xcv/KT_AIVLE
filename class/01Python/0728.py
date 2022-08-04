# -*- coding: utf-8 -*-

#미니실습 1
def min4(a, b, c, d):
    arr=[]
    arr.append(a)
    arr.append(b)
    arr.append(c)
    arr.append(d)
    
    return max(arr)

#미니실습 2
def waterPay(company, usage):
    ret=0
    
    if(company=='A'):
        ret=usage*100
    elif(company=='B'):
        if(usage<=50):
            ret=usage*150
        else:
            ret=50*150+(usage-50)*75
    else:
        ret=-1
        
    return ret

#미니실습 3
def func():
    a = int(input('a : '))
    b = int(input('b : '))
    if a > b:
        a, b = b, a
    
    sum = 0
    
    for i in range(a, b):
        if i < b:
            print(f'{i} + ', end='')
        sum +=i
        
    print(f'{b} = ',end='')
    sum+=b

    print(sum)

    return

#미니실습 4
def rectangle(area):
    for i in range(1,area+1):
        if(area%i==0):
            print(f'{i} x {area//i} = {area}')
            
    return

#실습 1
def electricPay(usage):
    sum=0
    
    if usage<100:
        sum=410
        sum+=60.7*usage
    elif usage<=200:
        sum=910
        sum+=100*60.7+125.9*(usage-100)
    else:
        sum=1600
        sum+=100*(60.7+125.9)+187.9*(usage-200)
        
    sum=int(sum*(1+0.1+0.037))
    
    print(f'요금 : {sum}')
    
    return
    
#실습 2       
def plusminus(cnt):
    quo=cnt//2
    rem=cnt%2
    
    print('+-'*quo,end='')
    
    if(rem==1):
        print('+')
    else:
        print()
    
    return

#실습 3
def not8():
    for i in range(1,13):
        if(i!=8):
            print(i,end=' ')
    print()
    
    return
    
#실습 4
def rootpwr(n):
    root=1
    
    while root**2<=n:
        for pwr in [2,3,4,5]:
            if(root**pwr==n):
                print(f'({root}, {pwr})')
                break
            
        root+=1
        
    return

#추가실습 0     
def gcd(a, b):
    if(a<b):
        a,b=b,a
    
    while(b!=0):
        a,b=b,a%b
        
    return a

#추가실습 1
def lcm(a,b):
    num=int(a*b/gcd(a,b))
    
    return num

#추가실습 2
def cycle(num):
    old=num
    cnt=1
    
    while True :
        right1=old%10
        sum=old//10+old%10
        right2=sum%10
        
        new=right1*10+right2
        
        if(num==new):
            break
        else:
            old=new
            cnt+=1    
    
    return cnt

#추가실습 3
def lr(left, right):
    # 1<=left<=right<=1000
    sum = 0
    
    for i in range(left, right+1):
        flag=False
        
        for j in range(1, 32):
            if(j*j>i):
                break
            elif(j*j==i):
                flag=True
    
        if flag:
            sum-=i
        else:
            sum+=i
    
    return sum

#추가실습 4
def Max(data):
    # data는 숫자로 이루어진 문자열
    arr=tuple(data)
    sum=0
    
    for i in arr:
        num=int(i)
        
        if(sum==0):
            if(num!=0):
                sum=num
        else:
            if(num==0 or num==1):
                sum+=num
            else : 
                sum*=num    
    
    return sum

#추가실습 5
def lucky(num):
    # num은 항상 자릿수가 짝수인 숫자로 이루어진 문자열
    arr = tuple(num)
    length = len(arr)
    cnt=int(length/2)
    
    left=0
    right=0
    
    for i in range(0,cnt):
        left+=int(arr[i])
        
    for i in range(cnt, length):
        right+=int(arr[i])
        
    if(left==right):
        print("LUCKY")
    else:
        print("READY")
    
    return 

#추가실습 6
def square(num):
    base=1
    result=-1
    
    while True:
        if(base**2>num):
            break
        elif(base**2==num):
            result=(base+1)**2
            break
        
        base+=1
    
    return result

##################################################

def main():
    # 문제별 함수에 맞게 파라미터 선언 및 입력 필요함
    num=int(input('정수입력 : '))
    result=square(num)
    print(f'결과 : {result}')

    
main()

##################################################