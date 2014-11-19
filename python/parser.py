import gzip
import simplejson

def parse(filename):
  f = gzip.open(filename, 'r')
  entry = {}
  for l in f:
    l = l.strip()
    colonPos = l.find(':')
    if colonPos == -1:
      yield entry
      entry = {}
      continue
    eName = l[:colonPos]
    rest = l[colonPos+2:]
    entry[eName] = rest
  yield entry

filename = '/oasis/projects/nsf/csd181/arapat/data/Electronics.txt.gz'
counter = 0
for e in parse(filename):
    print e['review/text'].replace('\n', ' ')
    counter += 1
    if counter == 10:
        break
