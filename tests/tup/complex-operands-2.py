a = (1, 2, 3, True)
b = (a,)
if a[1] > b[0][0] and a[3]:
    print(1)
else:
    print(0)
