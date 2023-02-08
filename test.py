import time
def myfun(i, j):
    time.sleep(1)
    a = i + j
    print(a)

# t = time.time()
# for _ in range(5):
#     myfun()
# print(time.time() - t)

from threading import Thread
t = time.time()
ths = []
for i in range(5):
    for j in range(5):
        th = Thread(target = myfun, args=(i,j))
        th.start()
        ths.append(th)
for th in ths:
    th.join()
print(time.time() - t)