a = (1, True and False, (3, 4))
b = a[0] + a[2][0]
print(b)
if a[1]:
    print(0)
else:
    print(1)
