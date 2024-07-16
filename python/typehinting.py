import os

# Build a type hinting helper for libbifrost_generated.py
def build_typehinting(filename):
    enums = {'status':   {},
             'space':    {},
             'dtype':    {},
             'capture':  {},
             'io':       {},
             'whence':   {},
             'reduce':   {}}
    
    with open(filename, 'r') as fh:
        for line in fh:
            if line.startswith('BF_'):
                for tag in enums.keys():
                    if line.startswith(f"BF_{tag.upper()}_"):
                        name, value = line.split('=', 1)
                        
                        name = name.strip().rstrip()
                        value = value.strip().rstrip()
                        enums[tag][name] = value
                        
                        if tag == 'space':
                            name = name.replace('BF_SPACE_', '')
                            enums[tag][name.lower()] = value
                        elif tag == 'io':
                            name = name.replace('BF_IO_', '')
                            enums[tag][name.lower()] = value
                        elif tag == 'reduce':
                            name = name.replace('BF_REDUCE_', '')
                            name = name.replace('POWER_', 'pwr')
                            enums[tag][name.lower()] = value
                        break
                        
    outname = filename.replace('generated', 'typehints')
    with open(outname, 'w') as fh:
        fh.write(f"""
\"\"\"
Type hints generated from {filename}

Do not modify this file.
\"\"\"

import enum

""")
        for tag in enums.keys():
            fh.write(f"class BF{tag}_enum(enum.IntEnum):\n")
            for key,value in enums[tag].items():
                fh.write(f"    {key} = {value}\n")
            fh.write("\n")
