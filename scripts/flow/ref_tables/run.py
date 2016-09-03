### Use to generate all tables
## substitute cliente and product with their low_ram version if needed

for ref in ['products', 'cliente', 'agencia', 'cliente', 'ruta']:
    print("Build reference table: {}".format(ref))
    with open("create_{}_w_agg.py".format(ref)) as f:
        code = compile(f.read(), "create_{}_w_agg.py".format(ref), 'exec')
        exec(code)