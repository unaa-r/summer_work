import itertools

As = [2500,5000,7500]
bands = [0.5,0.35,0.1]

with open("aug8params.txt", "w") as f:
    for combo in itertools.product(As, bands):
        f.write(" ".join(map(str, combo)) + "\n")
