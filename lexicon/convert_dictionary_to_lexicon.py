import os
import sys

if __name__ == "__main__":
    dict_file = sys.argv[1]
    out_dir = sys.argv[2]
    out_file = os.path.join(out_dir, "lexicon-"+os.path.basename(dict_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(dict_file, "r", encoding="utf-8") as f_in, open(out_file, "w+", encoding="utf-8") as f_out:
        data = f_in.readlines()
        for line in data:
            line = line.strip()
            if line.startswith("ngh"):
                f_out.write(line+"\t"+line[:3]+" "+line[3:]+"\n")
            elif line.startswith("ch") or line.startswith("gh") or line.startswith("gi") or line.startswith("kh") or line.startswith("ng") \
                or line.startswith("nh") or line.startswith("ph") or line.startswith("qu") or line.startswith("th") or line.startswith("tr") \
                :
                f_out.write(line+"\t"+line[:2]+" "+line[2:]+"\n")
            elif line.startswith("b") or line.startswith("c") or line.startswith("d") or line.startswith("đ") or line.startswith("g") or line.startswith("h")\
                or line.startswith("k") or line.startswith("l") or line.startswith("m") or line.startswith("n") or line.startswith("p") \
                or line.startswith("r") or line.startswith("s") or line.startswith("t") or line.startswith("v") or line.startswith("x"):
                f_out.write(line+"\t"+line[:1]+" "+line[1:]+"\n")
            else:
                f_out.write(line+"\t"+line+"\n")