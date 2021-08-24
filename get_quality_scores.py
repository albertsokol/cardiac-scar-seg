import json
import csv

if __name__ == '__main__':
    result = {}

    with open('/media/y4tsu/ml-fast/cmr/CMRImages2_certainty.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            if line[0].startswith('N'):
                curr = line[0]
                result[curr] = [int(line[2])]
                i = 1
            else:
                result[curr] += [int(line[2])]
                i += 1

    with open('quality_scores.json', 'w') as f:
        f.write(json.dumps(result, indent=2))
