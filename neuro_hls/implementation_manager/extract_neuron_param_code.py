def extract_neuron_param_code(mat, num_tabs = 0):

    if len(mat.shape) == 1:

        s = "\t" * num_tabs + "{"

        for i in range(mat.shape[0]):
            
            s += f"{mat[i]:.10f}"

            if i < mat.shape[0] - 1:
                s += ", "

        return s + "}"
    
    s = "\t" * num_tabs + "{"

    for i in range(mat.shape[0]):

        s += "\n" + extract_neuron_param_code(mat[i], num_tabs + 1)

        if i < mat.shape[0] - 1:
            s += ",\n"

    s += "\n" + "\t" * num_tabs + "}"
    return s
