import sys
import csv

file_name = sys.argv[1]
file_name_to_be_created = sys.argv[2]
columns_to_be_kept = sys.argv[3:]

with open(file_name, 'r') as f_read:
    reader = csv.reader(f_read)
    header = next(reader)

    indices = [header.index(column) for column in columns_to_be_kept]

    if 'score' in columns_to_be_kept:
        score_flag = True
    else:
        score_flag = False

    with open(file_name_to_be_created, 'w') as f_write:

        writer = csv.writer(f_write, delimiter=',', quotechar='"')
        writer.writerow(columns_to_be_kept)

        for row in reader:
            # we need to convert the score to a 0-5 scale from a 0-10 scale and round to integer
            if score_flag:
                if row[indices[-1]] != '':
                    row[indices[-1]] = str(round(float(row[indices[-1]])/2))
                else:
                    row[indices[-1]] = '0'
            writer.writerow([row[index] for index in indices])
    f_write.close()
f_read.close()
