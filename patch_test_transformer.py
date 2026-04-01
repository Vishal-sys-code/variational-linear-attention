with open("tests/test_transformer.py", "r") as f:
    code = f.read()

code = code.replace("VLATransformerBlock", "LRATransformerBlock")
code = code.replace("VLATransformer", "LRAModel")

with open("tests/test_transformer.py", "w") as f:
    f.write(code)
print("Patched!")
