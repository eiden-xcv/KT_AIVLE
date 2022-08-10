## 입력 받기
0. sys.stdin.readline() vs input()
sys.stdin.readline() 개행문자도 받기 때문에 sys.stdin.readline().strip()으로 받아야함.
속도 차이가 크기에 sys.stdin.readline()으로 받아주는게 좋음.
```
import sys

s1 = sys.stdin.readline()
s2 = input()

arr=[s1, s2]

print(arr)
==========================
>> ['hello\n', 'hello']
```
1. 정수/실수로 변환
```
n = int(input()) / float(intput()) 
```
2. 공백으로 구분된 문자를 리스트로 받기
```
arr = input().split
```
3. 공백으로 구분된 숫자를 정수로 변환하여 변수에 저장
```
a, b = map(int, input().split())
```
4. 공백으로 구분된 수열 리스트로 받기
```
arr = list(map(int, input().split()))
```
5. 이어진 문자를 하나씩 리스트에 저장
```
arr = list(input())
```
6. 이어진 숫자를 하나씩 리스트에 저장
```
arr = list(map(int, input()))
```
7. 여러 문자열 리스트에 저장
```
arr = [input() for _ in range(N)]
```
8. 2차원 배열 받기
```
arr = []
for _ in range(N):
  arr.append(list(map(int, input().split())))

or

arr = [list(map(int, input().split())) for _ in range(N)]
```
