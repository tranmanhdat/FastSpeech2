import sys

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        data = f.readlines()
        symbols = []
        for line in data:
            line = line.strip()
            elements = line.split("\t")
            sub_element = elements[1].split(" ")
            for element in sub_element:
                if element not in symbols:
                    symbols.append(element)
        symbols.sort()
        with open(sys.argv[2], "w+") as f:
            f.write(str(symbols))