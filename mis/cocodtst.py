import json

coco_path = r'C:\Users\gtx97\Downloads\captions_val2014.json'

coco_val = json.load(open(coco_path, 'r'))
print(type(coco_val))
print()

print('keys')
print(coco_val.keys())
print()

print('info')
print(coco_val['info'])
print()

print('images')
print(type(coco_val['images']))
print(coco_val['images'][0])
print(len(coco_val['images']))
print()

print('licenses')
print(type(coco_val['licenses']))
print(len(coco_val['licenses']))
print(coco_val['licenses'])
print()

print('annotations')
print(type(coco_val['annotations']))
print(len(coco_val['annotations']))
print(coco_val['annotations'][0])
print(coco_val['annotations'][1])


