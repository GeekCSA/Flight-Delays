import csv

with open('Data_set/demo_flights.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(row[4])
        else:
            print(row[4])
        line_count += 1

    print('Processed {line_count} lines.')