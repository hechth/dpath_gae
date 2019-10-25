def dummy_function(x):
    print(x)
    return x

file = "def fun(*args): \n  return args"
func = "fun(1,2,3)"

def execute(func, file):
    program = file + "\nresult = " + func
    local = {}
    exec(program, local)
    return local['result']