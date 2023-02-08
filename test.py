import time
def myfun():
    time.sleep(1)
    a = 1 + 1
    print(a)

t = time.time()
for _ in range(5):
    myfun()
print(time.time() - t)

from threading import Thread
t = time.time()
ths = []
for _ in range(5):
    th = Thread(target = myfun)
    th.start()
    ths.append(th)
for th in ths:
    th.join()
print(time.time() - t)