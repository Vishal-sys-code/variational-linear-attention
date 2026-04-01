with open("tests/test_transformer.py", "r") as f:
    code = f.read()

code = code.replace("logits = model(x)", "logits = model(x, pool=False)")
code = code.replace("logits = model(x, pool=False)", "logits = model(x, pool=False)") # Make sure not duplicated

with open("tests/test_transformer.py", "w") as f:
    f.write(code)
print("Patched!")
