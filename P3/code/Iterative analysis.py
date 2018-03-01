
#解析OSM文件，并计算其中的tags的数量
import xml.etree.cElementTree as ET
import pprint

OSMFILE = "sample.osm"
def count_tags(filename):
    tags = {}
    for event, elem in ET.iterparse(filename):
        if elem.tag in tags: 
            tags[elem.tag] += 1
        else:
            tags[elem.tag] = 1
    return tags
    
pprint.pprint(count_tags(OSMFILE))
