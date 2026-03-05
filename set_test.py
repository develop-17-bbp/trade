import json
s = set()
try:
    s.add({'a':1})
except Exception as e:
    print('add_error', e)

try:
    print(json.dumps(s))
except Exception as e:
    print('dumps_error', e)
