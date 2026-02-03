import numpy as np

matriz = np.random.randint(0, 10, size=(1, 2, 3, 5))
print(f"MAT: {matriz}\n\n")

def extract(mat, num_tabs = 0):

    if len(mat.shape) == 1:

        s = "\t" * num_tabs + "{"

        for i in range(mat.shape[0]):
            
            s += f"{mat[i]:.4f}"

            if i < mat.shape[0] - 1:
                s += ", "

        return s + "}"
    
    s = "\t" * num_tabs + "{"

    for i in range(mat.shape[0]):

        s += "\n" + extract(mat[i], num_tabs + 1)

        if i < mat.shape[0] - 1:
            s += ",\n"

    s += "\n" + "\t" * num_tabs + "}"
    return s
    
s = extract(matriz)
print(s)

