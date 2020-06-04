import csv

# parse the counters and scenario information to visualize in the figures


def parseNS():

    file = "../../Datasets/Dynamism/Network Size/dyn_ns_om.csv"

    data = []
    with open(file, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=';')
        for row in csv_reader:
            data.append(int(row['Value']))
    return data


def parseCNT(case):
    file = ''
    if case == "ps":
        file = "../../Datasets/Dynamism/Payload Size/dyn_ps_cnt.csv"
    if case == "sf":
        file = "../../Datasets/Dynamism/Spreading Factor/dyn_sf_cnt.csv"

    data = dict()

    with open(file, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=';')
        line_count = 0
        for row in csv_reader:
            line = str(row['Series'])
            linesplit = line.split(' ')[2:]
            key = ''
            if len(linesplit) == 1:
                key = linesplit[0][:-1]
            else:
                key = linesplit[0][:-1] + "_" + linesplit[2]
                key = key[21:]
            key = key.replace("/", '_').replace(",",'')
            if key in data:
                times = data[key]
                value = int(row['Value'])
                data[key].append(value)
                print(key)
            else:
                data[key] = []

        return data

def parseSensorData(case):
    file = ''
    if case == "ps":
        file = "../../Datasets/Dynamism/Payload Size/dyn_ps_ps.csv"
    if case == "sf":
        file = "../../Datasets/Dynamism/Spreading Factor/dyn_sf_sf.csv"

    data = dict()

    with open(file, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=';')
        line_count = 0
        for row in csv_reader:
            line = str(row['Series'])
            linesplit = line.split(' ')[2:]
            key = ''
            if len(linesplit) == 1:
                key = linesplit[0][:-1]
            else:
                key = linesplit[0][:-1] + "_" + linesplit[2]
                key = key[21:]
            key = key.replace("/", '_').replace(",",'')
            if key in data:
                times = data[key]
                value = int(row['Value'])
                data[key].append(value)
            else:
                data[key] = []

        return data
