import sys
from text.symbols import symbols

if __name__=="__main__":
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()
        i = 0
        while(i<len(lines)):
            if lines[i].__contains__("item [2]:"):
                break
            i += 1
        i = i + 6
        while(i<len(lines)):
            symbol = lines[i+3].split("\"")[-2]
            if symbol not in symbols:
                print(symbol)
            i = i + 4