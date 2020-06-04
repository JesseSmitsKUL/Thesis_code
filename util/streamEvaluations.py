import csv


# this file showed to strong alternating behavior of the wireless mesh arrival times

makePlots = True
savePlots = False
saveLocation = "./Visuals/ExpSmoothing/"
filePath = "../Datasets/rats_inc.csv"
prefix = "/ExpSm_alpha_125"



def streamTrends(stream):
    alternating = 0
    followingTrend = 0
    direction = 'up' if stream[0] < stream[1] else 'down'
    last = stream[1]
    for packet in stream[2:]:
        newDirection =  'up' if last < packet else 'down'
        if newDirection == direction:
            followingTrend += 1
        else:
            alternating += 1
        direction = newDirection
        last = packet
    return (alternating,followingTrend)



data = dict()
with open(filePath,'r', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter=';')
    line_count = 0
    for row in csv_reader:
        line = str(row['Series'])

        linesplit = line.split(' ')[2:]


        key = linesplit[0][:-1] + "_" + linesplit[2]
        key = key[21:]
        if key in data:
            times = data[key]
            value = float(row['Value'])
            if len(times) > 0 and value == times[-1]:
                value += 0.001
            data[key].append(value)
        else:
            data[key] = []
        #print(linesplit)
    #print(data)

datakeys = [*data]
datakeys.sort()

print(datakeys)



for key in datakeys:
    stream = data[key]
    alt, fol = streamTrends(stream)
    print('trends alternating: ', alt, ' trends following', fol , " ratio: ", fol/alt)



