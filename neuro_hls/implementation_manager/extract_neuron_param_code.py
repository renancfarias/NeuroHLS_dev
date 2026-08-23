def extract_neuron_param_code(mat, num_tabs = 0, value_format = ".10f"):

    if len(mat.shape) == 1:

        s = "\t" * num_tabs + "{"

        for i in range(mat.shape[0]):
            
            s += format(mat[i], value_format)

            if i < mat.shape[0] - 1:
                s += ", "

        return s + "}"
    
    s = "\t" * num_tabs + "{"

    for i in range(mat.shape[0]):

        s += "\n" + extract_neuron_param_code(
            mat[i], num_tabs + 1, value_format
        )

        if i < mat.shape[0] - 1:
            s += ",\n"

    s += "\n" + "\t" * num_tabs + "}"
    return s
