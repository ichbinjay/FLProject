fp1 = open(r"C:\Users\hp\PycharmProjects\pythonProject\Book1.csv", "r")
fp2 = open(r"C:\Users\hp\PycharmProjects\pythonProject\Book2.csv", "w")

for line in fp1:
    words = line.split(",")
    if words[-1] == "h0":
        words[-1] = "1"
        fp2.write(",".join(words)+"\n")
    elif words[-1] == "h1":
        print("here")
        words[-1] = "0"
        fp2.write(",".join(words)+"\n")

fp1.close()
fp2.close()